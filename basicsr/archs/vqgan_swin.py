import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_swinir import RSTB

from basicsr.utils.registry import ARCH_REGISTRY

from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (d, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


def normalize(in_channels):
    if in_channels >=32:
        gp = 32
    else:
        gp = in_channels // 4
    return torch.nn.GroupNorm(num_groups=gp, num_channels=in_channels, eps=1e-6, affine=True)


@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, heads=8, mid_ch=64, norm_type='gn'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_ch = heads * mid_ch
        self.heads = heads

        self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)

        self.qkv = nn.Conv2d(in_channels, mid_ch * heads * 3, 1, 1, 0)
        self.proj_out = nn.Conv2d(mid_ch * heads, in_channels, 1, 1, 0)
    
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q, k, v = self.qkv(h_).chunk(3, dim=1)

        # compute attention
        b,c,h,w = q.shape
        q, k, v = map(lambda t: rearrange(t, 'b (n d) h w -> b (h w) n d', n = self.heads), (q, k, v))
        v = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        v = rearrange(v, 'b (h w) n d -> b (n d) h w', h=h, w=w) 
        h_ = self.proj_out(v)

        return x+h_


class SwinLayers(nn.Module):
    def __init__(self, embed_dim, swin_dim=256, 
                blk_num=1, blk_depth=6, window_size=8,
                num_heads=8, input_resolution=(32, 32), **kwargs):
        super().__init__()
        self.in_dim = embed_dim
        self.swin_dim = swin_dim

        self.conv_in = nn.Conv2d(self.in_dim, self.swin_dim, 3, 1, 1)
        self.conv_out = nn.Conv2d(self.swin_dim, self.in_dim, 3, 1, 1)

        self.swin_blks = nn.ModuleList()
        for i in range(blk_num):
            layer = RSTB(swin_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x_in):
        x = self.conv_in(x_in)

        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.conv_out(x)

        return x + x_in


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class Encoder(nn.Module):
    def __init__(self, *, in_channels, nf, emb_dim, ch_mult, 
                num_res_blocks=2, resolution, swin_blk_num=1, 
                swin_blk_depth, swin_dim, swin_window=8, global_blks='swin', **swin_opts):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
    
        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = (curr_res[0]//2, curr_res[1]//2)

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        if global_blks == 'swin':
            blocks.append(SwinLayers(embed_dim=block_in_ch,swin_dim=swin_dim,blk_num=swin_blk_num,
                                blk_depth=swin_blk_depth,window_size=swin_window,
                                input_resolution=curr_res, **swin_opts))
        elif global_blks == 'mhsa':
            blocks += [AttnBlock(block_in_ch)]
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        if global_blks == 'mhsa':
            blocks += [AttnBlock(block_in_ch)]

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, out_feats=False, txt_feat=None):
        if out_feats:
            feats = {}
        for block in self.blocks:
            if block.__class__.__name__ == 'BasicBlock':
                x = block(x, features_text=txt_feat)
            else:
                x = block(x)
            if out_feats:
                feats[x.shape[-1]] = x.clone()
        if out_feats:
            return x, feats
        else:
            return x


class Decoder(nn.Module):
    def __init__(self, *, nf, emb_dim, ch_mult, res_blocks=2, resolution, 
                        swin_blk_num=1, swin_blk_depth, swin_dim, swin_window=8, global_blks='swin', **kwargs):
        super().__init__()
        self.nf = nf #64
        self.ch_mult = ch_mult #[1, 2, 2, 4, 4, 8]
        self.num_resolutions = len(self.ch_mult) #5
        self.num_res_blocks = res_blocks #2
        self.resolution = resolution #512
        self.in_channels = emb_dim #256
        self.out_channels = 3 #3
        block_in_ch = int(nf * ch_mult[-1]) #512
        curr_res = (self.resolution[0] // 2**(self.num_resolutions-1), \
            self.resolution[1] // 2**(self.num_resolutions-1))

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        if global_blks == 'swin':
            blocks.append(SwinLayers(embed_dim=block_in_ch,swin_dim=swin_dim,blk_num=swin_blk_num,
                                blk_depth=swin_blk_depth,window_size=swin_window,
                                input_resolution=curr_res,))
        elif global_blks == 'mhsa':
            blocks += [AttnBlock(block_in_ch)]
        
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        if global_blks == 'mhsa':
            blocks += [AttnBlock(block_in_ch)]

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = int(self.nf * self.ch_mult[i])

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = (curr_res[0]*2, curr_res[1]*2)

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, residual_32=None, residual_64=None):
        # feats_dict = [9, 12, 15]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 9 and residual_32 is not None: #32 after frist res block
                x = x + residual_32
            if i ==12 and residual_64 is not None: #64
                x = x + residual_64
        return x       


@ARCH_REGISTRY.register()
class VQModel(nn.Module):
    def __init__(self, 
                 ddconfig, 
                 n_embed,
                 embed_dim,
                 ckpt_path = None,
                 beta=0.25, 
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.generator = self.decoder
        
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=beta, sane_index_shape=True)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'params_ema' in ckpt:
                ckpt = ckpt['params_ema']
            else:
                ckpt = ckpt['params']

            self.load_state_dict(ckpt, strict=True)
    
    def encode(self, x, txt_feat=None):
        x = x.clamp(0, 1)
        x = x * 2 - 1
        h = self.encoder(x, txt_feat=txt_feat)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def encode_to_prequant(self, x, txt_feat=None):
        x = x.clamp(0, 1)
        x = x * 2 - 1
        h = self.encoder(x, txt_feat=txt_feat)
        return h

    def decode(self, quant, res_feats=None):
        dec = self.decoder(quant, res_feats)
        dec = (dec + 1) / 2
        dec = dec.clamp(0, 1)
        return dec
    
    def encode_to_indices(self, input, txt_feat=None):
        quant, diff, (_,_,ind) = self.encode(input, txt_feat)
        return quant, ind
    
    def feat_to_indices(self, feat):
        quant, diff, (_,_,ind) = self.quantize(feat)
        return ind
    
    def decode_indices(self, indice, res_feats=None):
        b, _, h, w = indice.shape
        shape = [b, h, w, self.embed_dim]
        indice = indice.to(self.quantize.embedding.weight.data.device)
        quant_b = self.quantize.get_codebook_entry(indice, shape)
        dec = self.decode(quant_b, res_feats)
        return dec
    
    def decode_onehot(self, onehot_idx):
        b, c, h, w = onehot_idx.shape
        codebook = self.quantize.embedding.weight.data
        onehot_idx = onehot_idx.permute(0, 2, 3, 1).to(codebook)
        quant_feat = onehot_idx @ codebook 
        dec = self.decode(quant_feat.permute(0, 3, 1, 2))
        return dec
    
    def test(self, input):
        # padding to multiple of wsz
        wsz = 64
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        rec, _ = self.forward(input)
        rec = rec[:, :, :h_old, :w_old]
        return rec
    
    def forward(self, input, return_pred_indices=False, txt_feat=None):
        quant, diff, (_,_,ind) = self.encode(input, txt_feat)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff



