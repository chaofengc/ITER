import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

from .iter_utils import ResBlock, class_balanced_xentropy
from .demask_arch import UNetModel, SwinModel, TransformerModel, SwinLayers

from .vqgan_swin import VQModel
from .demask_arch import TransformerModel

from einops import rearrange


def safe_interpolate(x, *args, **kwargs):
    """Use fp32 for interpolation, might be useful when using mixed precision training.
    """
    return F.interpolate(x.float(), *args, **kwargs).to(x)


class TokenRefine(nn.Module):

    def __init__(self, n_e, T=8, temp_token=1.0, temp_mask=1.0, gamma_type='cosine', init_mask_threshold=0.5, use_adaptive_inference=True, demask_net='swin'):
        super().__init__()
        self.n_e = int(n_e)
        self.n_embed = int(n_e)

        self.T = T
        self.temp_token = temp_token
        self.temp_mask = temp_mask
        self.init_mask_threshold = init_mask_threshold 
        self.gamma_type = gamma_type
        self.use_adaptive_inference = use_adaptive_inference

        if demask_net == 'unet':
            mid_ch = 128
            ch_mult = (1, 2, 4, 8)
            self.token_crit_net = UNetModel(mid_ch, n_e, 2, ch_mult=ch_mult, num_res_blocks=2)
            self.token_refine_net = UNetModel(mid_ch, n_e + 1, n_e, ch_mult=ch_mult, num_res_blocks=2)
        elif demask_net == 'swin':
            mid_ch = 256
            self.token_crit_net = SwinModel(mid_ch, n_e, 2)
            self.token_refine_net = SwinModel(mid_ch, n_e + 1, n_e)
        elif demask_net == 'transformer':
            mid_ch = 512
            self.token_crit_net = TransformerModel(mid_ch, n_e, 2)
            self.token_refine_net = TransformerModel(mid_ch, n_e + 1, n_e)
        
    @torch.no_grad() 
    def max_onehot(self, x, dim=1):
        c = x.shape[dim]
        x = x.argmax(dim=dim)
        x = F.one_hot(x, c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()
    
    @torch.no_grad() 
    def sample_t(self, mask_logits, k, prev_mask=None, temp=1.0):
        k = max(1, k)
        b, c, h, w = mask_logits.shape
        mask_logits = mask_logits / temp
        mask_logits = mask_logits.float().softmax(dim=1)

        mask_prob = mask_logits[:, [1]]

        # only sample from the left masked region
        if prev_mask is not None:
            mask_prob = mask_prob * (1 - prev_mask)
            k = k - prev_mask.flatten(1).sum(dim=-1)[0].item()
            k = max(1, k)

        mask_prob = mask_prob.reshape(b, h * w)
        mask_out = torch.zeros_like(mask_prob)

        mask_ind = torch.multinomial(mask_prob + 1e-8, int(k))
        for i in range(b):
            mask_out[i, mask_ind[i]] = 1
        mask_out = mask_out.reshape(b, 1, h, w)

        if prev_mask is not None:
            mask_out = mask_out + prev_mask 
        
        return mask_out
    
    @torch.no_grad() 
    def sample_k(self, sample_rate, gt_mask_wosample):
        b, _, h, w = gt_mask_wosample.shape
        rand_prob = torch.rand(b, 1, h, w) - gt_mask_wosample.float().cpu()
        sample_k = (sample_rate * h * w).int()
        sample_k = sample_k.clamp(min=1) 
        topk = torch.topk(rand_prob.reshape(b, h * w), sample_k.max(), dim=-1)[0]
        mask = rand_prob >= topk[range(b), sample_k.long() - 1].reshape(b, 1, 1, 1)
        return mask.to(gt_mask_wosample).float()
    
    @torch.no_grad() 
    def matrix_onehot(self, x, num_classes):
        x = F.one_hot(x[:, 0].long(), num_classes)
        return x.permute(0, 3, 1, 2).contiguous().float()
    
    @torch.no_grad()
    def sample_categorial(self, token_logits, temp=1.0):
        b, c, h, w = token_logits.shape
        token_logits = token_logits.float() / temp
        token = torch.distributions.categorical.Categorical(logits=token_logits.permute(0, 2, 3, 1).flatten(0, 2)).sample()
        token = self.matrix_onehot(token.reshape(b, 1, h, w), self.n_e)
        return token
    
    def token_refine(self, init_onehot_token, sampled_token, token_mask, T=8, temp_token=1.0, temp_mask=1.0):

        token_mask = token_mask[:, [0]]

        T = self.T
        temp_token = self.temp_token
        temp_mask = self.temp_mask
        gamma_type = self.gamma_type

        token = token_logits = init_onehot_token
        mask = token_mask.float()

        self.refine_steps = 1 if self.training else T

        prev_mask = None
        log_img_freq = 1 
        token_logits = None
        init_mask = token_mask

        if not self.training and self.use_adaptive_inference:
            mask = self.token_crit_net([token, init_onehot_token])
            init_mask_binary = mask.argmax(dim=1, keepdim=True).float()
            tmp_mask = mask.float().softmax(dim=1)[:, [1]]
            init_mask_binary = (tmp_mask >= self.init_mask_threshold).float()
            # initialization
            prev_mask = init_mask_binary
            mask = init_mask_binary
            sampled_token = init_onehot_token * init_mask_binary

            init_mask = mask
        elif self.training:
            with torch.no_grad():
                init_mask = self.token_crit_net([init_onehot_token, init_onehot_token])
                init_mask = init_mask.argmax(dim=1, keepdim=True).float()
        
        out_tokens = [] 
        for i in range(self.refine_steps):
            log_img = i % log_img_freq == 0 or i == self.refine_steps - 1
            if log_img:
                out_tokens.append(token.argmax(dim=1, keepdim=True))

            if not self.training:
                num_gt_samples = int((1 - self.gamma_func(torch.Tensor([i / T]), gamma_type)) * mask.shape[-1] * mask.shape[-2])
                if self.use_adaptive_inference:
                    if num_gt_samples < init_mask_binary.sum().item():
                        # print(f'Skip step {i}')
                        if log_img:
                            self.eval_refine_masks.append(init_mask_binary)
                        continue
                        
                if mask.shape[1] > 1:
                    mask = self.sample_t(mask, num_gt_samples, prev_mask, temp=temp_mask) 
                    prev_mask = mask

                # prev_mask = mask
                if log_img:
                    self.eval_refine_masks.append(mask)

            tokens = [torch.cat([init_onehot_token, 1 - init_mask], dim=1), torch.cat([sampled_token * mask, 1 - mask], dim=1)]

            token_logits = self.token_refine_net(tokens)

            if i < self.refine_steps:
                sampled_token = self.sample_categorial(token_logits, temp=temp_token)
                token = sampled_token
                mask = self.token_crit_net([token, init_onehot_token])

        if self.training:
            return token_logits
        else:
            out_tokens.append(token.argmax(dim=1, keepdim=True))
            return out_tokens[-T:], token_logits

    def gamma_func(self, r, mode="linear"):
        if mode == "linear":
            return 1 - r
        elif mode == "cosine":
            return torch.cos(r * math.pi / 2)
        elif "pow" in mode:
            exponent = float(mode.replace("pow", ""))
            return 1. - r ** exponent
        elif "exp" in mode:
            exponent = float(mode.replace("exp", ""))
            return (exponent - 1) / exponent * (1. - exponent ** (r - 1))
        else:
            raise NotImplementedError
        
    def forward(self, init_token, gt_indices=None, features_text=None):
        init_token = self.matrix_onehot(init_token, self.n_e) 
        b, c, h, w = init_token.shape
        init_token_org = init_token.float()

        codebook_loss = 0
        self.ret_masks = []
        self.eval_refine_masks = []
        # if train, sampling prediction tokens 
        if self.training:
            # show initial token mask 
            with torch.no_grad():
                gt_mask_wosample = (gt_indices == init_token.argmax(dim=1, keepdim=True))
            
            init_pred_mask_logits = self.token_crit_net([init_token, init_token_org])
            codebook_loss += class_balanced_xentropy(init_pred_mask_logits.permute(0, 2, 3, 1).flatten(0, 2), gt_mask_wosample.flatten().long()) 

            init_pred_mask = init_pred_mask_logits.argmax(dim=1, keepdim=True)
            self.ret_masks.append(init_pred_mask)
            self.ret_masks.append(gt_mask_wosample)

            with torch.no_grad():
                sample_rate = self.gamma_func(torch.rand(gt_indices.shape[0]), 'cosine')
                gt_onehot = F.one_hot(gt_indices[:, 0], self.n_embed).permute(0, 3, 1, 2).contiguous()
                sample_mask = self.sample_k(sample_rate, torch.zeros_like(gt_mask_wosample))
                sampled_token = gt_onehot * (1 - sample_mask)

        if gt_indices is not None:
            gt_mask = (gt_indices == init_token.argmax(dim=1, keepdim=True))

        if self.training:

            # self.ret_masks.append(1 - sample_mask)
            # self.ret_masks.append(gt_mask)

            refined_logits = self.token_refine(init_token, sampled_token, 1 - sample_mask, T=1)

            # mask prediction loss for refined S_t
            refined_token_for_mask = self.max_onehot(refined_logits)
            pred_mask_logits = self.token_crit_net([refined_token_for_mask, init_token_org])
            pred_mask = pred_mask_logits.argmax(dim=1, keepdim=True)
            gt_mask = (gt_indices == refined_token_for_mask.argmax(dim=1, keepdim=True))
            self.ret_masks.append(pred_mask)
            self.ret_masks.append(gt_mask)

            pred_mask_logits = pred_mask_logits.float()
            mask_loss = F.cross_entropy(pred_mask_logits.permute(0, 2, 3, 1).flatten(0, 2), gt_mask.flatten().long())
            codebook_loss += mask_loss

            # refined token loss
            weight = sample_mask.mean(dim=(1, 2, 3)).unsqueeze(1)
            refined_logits = refined_logits.float()
            refine_token_loss = F.cross_entropy(refined_logits.permute(0, 2, 3, 1).contiguous().flatten(0, 2), gt_indices.flatten(), reduction='none')
            refine_token_loss = refine_token_loss.reshape(b, -1)
            refine_token_loss = (refine_token_loss * weight).mean()

            codebook_loss += refine_token_loss
            refined_token = refined_logits.argmax(dim=1, keepdim=True)
        else:
            pred_mask_logits = self.token_crit_net([init_token, init_token_org])

            refined_token, refined_logits = self.token_refine(init_token, torch.zeros_like(init_token), torch.zeros_like(init_token))

            self.ret_masks += self.eval_refine_masks
        
        if self.training:
            return refined_logits, codebook_loss, self.ret_masks
        else:
            return refined_token, codebook_loss, self.ret_masks, refined_logits


class LQEncoder(nn.Module):
    def __init__(self, in_ch, feat_dim, out_dim, depth, chn=64) -> None:
        super().__init__()

        blks = [nn.Conv2d(in_ch, chn, 3, 1, 1)]

        in_ch = chn
        for _ in range(depth):
            out_ch = in_ch * 2
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                ResBlock(out_ch, out_ch),
                ResBlock(out_ch, out_ch),
            ]
            blks.append(nn.Sequential(*tmp_down_block))
            in_ch = out_ch
        
        blks.append(nn.Conv2d(in_ch, feat_dim, 1, 1, 0))
        blks.append(SwinLayers(embed_dim=feat_dim, swin_dim=feat_dim, blk_num=2, blk_depth=6))

        blks.append(nn.Conv2d(feat_dim, out_dim, 1, 1, 0))
        self.model = nn.Sequential(*blks)
    
    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class ITER(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 LQ_train_phase=None,
                 norm_type='gn',
                 act_type='silu',
                 scale_factor=4,
                 vqgan_opt=None,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]

        self.vqgan = VQModel(**vqgan_opt)

        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.LQ_train_phase = LQ_train_phase
        self.scale_factor = scale_factor if LQ_stage else 1

        # build LQ encoder 
        if LQ_stage:
            for n, p in self.vqgan.named_parameters():
                p.requires_grad = False
            self.vqgan.eval()

            codebook_emb_num = vqgan_opt['n_embed'] 
            self.token_refine_net = TokenRefine(codebook_emb_num)

            # lq images to token index
            feat_dim = vqgan_opt['embed_dim']
            self.lq_encoder = LQEncoder(3, feat_dim, codebook_emb_num, int(np.log2(8/scale_factor)))

    def set_sample_params(self, T, temp_token, temp_mask, init_mask_threshold, gamma_type='cosine'):
        self.token_refine_net.T = T
        self.token_refine_net.temp_token = temp_token
        self.token_refine_net.temp_mask = temp_mask
        self.token_refine_net.init_mask_threshold = init_mask_threshold
        self.token_refine_net.gamma_type = gamma_type 
        self.token_refine_net.use_adaptive_inference = True
            
    def encode_and_decode(self, input, gt_indices=None, return_inter_results=True, text=None):
        input_org = input

        if not self.LQ_stage:
            out_img, emb_loss, indices = self.vqgan(input, True) 
            return out_img, emb_loss, None, indices.unsqueeze(1), None

        total_loss = 0

        # lq images to token 
        lq_token_logits = self.lq_encoder(input)
        lq_tokens = lq_token_logits.argmax(dim=1, keepdim=True)

        if self.training:
            # pixel level backpropagation for better token prediction
            lq_tokens_train = F.gumbel_softmax(lq_token_logits.float(), dim=1, tau=1, hard=True)
            out_img = self.vqgan.decode_onehot(lq_tokens_train.long())

            if gt_indices is not None:
                # token classification loss
                lq_token_logits = rearrange(lq_token_logits.float(), 'b c h w -> (b h w) c')
                token_cls_loss = F.cross_entropy(lq_token_logits, gt_indices.flatten())
                total_loss += token_cls_loss

        ret_masks = None
        refined_img = None

        if self.LQ_train_phase == 'token_pred':
            if not self.training:
                with torch.no_grad():
                    out_img = self.vqgan.decode_indices(lq_tokens)
        else: 
            # token refine
            if self.training:
                refined_logits, refine_loss, ret_masks = self.token_refine_net(lq_tokens, gt_indices)

                # pixel level backpropagation for better token prediction
                refined_token = F.gumbel_softmax(refined_logits.float(), dim=1, tau=1, hard=True)
                refined_img = self.vqgan.decode_onehot(refined_token)
            else:
                refined_token, refine_loss, ret_masks, refined_logits = self.token_refine_net(lq_tokens, gt_indices)
                refined_img = None

            total_loss += refine_loss

            if not self.training:
                with torch.no_grad():
                    if return_inter_results:
                        out_img = []
                        for tk in refined_token:
                            tmp_out = self.vqgan.decode_indices(tk.long())
                            out_img.append(tmp_out)
                    else:
                        out_img = [self.vqgan.decode_indices(refined_token[-1].long())]

        return out_img, total_loss, refined_img, lq_tokens, ret_masks
    
    @torch.no_grad()
    def test_tile(self, input, tile_size=256, tile_pad=16):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile, _ = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output
    
    @torch.no_grad()
    def test(self, input, return_inter_results=False):
        input_org = input
        # padding to multiple of window_size * 8
        if self.LQ_stage:
            wsz = 8 // self.scale_factor * 8 
        else:
            wsz = 16 * 8 
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        outputs = self.encode_and_decode(input, return_inter_results=return_inter_results)

        out = None
        if return_inter_results:
            out_img = outputs[0]
            out_img = torch.cat(out_img, dim=2)
            out_mask = torch.cat(outputs[-1], dim=2)
            out_mask = safe_interpolate(out_mask, out_img.shape[2:]).repeat(1, 3, 1, 1)

            # get intermediate results
            out = torch.cat([out_img, out_mask], dim=-1) 
            out = rearrange(out, 'b c (s h) (m w) -> b c (m h) (s w)', s=8, m=2)

        output = outputs[0][-1] 
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]
    
        return output, out 
    
    @torch.no_grad()
    def test_iterative(self, input):
        # padding to multiple of window_size * 8
        input_org = input
        wsz = 8 // self.scale_factor * 8 
        _, _, h_old, w_old = input.shape
        h_pad = (wsz - h_old % wsz) % wsz
        w_pad = (wsz - w_old % wsz) % wsz
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        outputs = self.encode_and_decode(input)

        out_img = outputs[0]
        out_mask = torch.cat(outputs[-1], dim=2)
        out_mask = safe_interpolate(out_mask, out_img.shape[2:]).repeat(1, 3, 1, 1)

        out = torch.cat([out_img, out_mask], dim=-1) 

        out = rearrange(out, 'b c (s h) (m w) -> b c (m h) (s w)', s=8, m=2)

        return list(out.chunk(8, dim=-1))
    
    def forward(self, input, gt_indices=None, text=None):

        if gt_indices is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, codebook_loss, semantic_loss, indices, ret_masks = self.encode_and_decode(input, gt_indices)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, codebook_loss, semantic_loss, indices, ret_masks = self.encode_and_decode(input, text=text)

        return dec, codebook_loss, semantic_loss, indices, ret_masks
