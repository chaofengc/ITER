import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vqgan_swin import Downsample, Upsample, AttnBlock, ResBlock, normalize, SwinLayers

from .llama_transformer import ModelArgs, Transformer
from einops import rearrange


class TransformerModel(nn.Module):
    def __init__(
        self, ch, in_channels, out_ch
    ): 
        super().__init__()

        self.code_emb = nn.Conv2d(in_channels, ch, 1, 1, 0, bias=False)
        self.combine = nn.Conv2d(ch * 2, ch, 1, 1, 0, bias=False)

        config = ModelArgs(
            dim=ch,
            max_seq_len=(2048 // 8) ** 2,
        ) 
        self.model = Transformer(config)

        self.conv_out = nn.Sequential(
            normalize(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 1, 1, 0)
        )

    def forward(self, x):
        if isinstance(x, list):
            code_emb1 = self.code_emb(x[0])
            code_emb2 = self.code_emb(x[1])
            x = self.combine(torch.cat([code_emb1, code_emb2], dim=1)) 
        else:
            x = self.code_emb(x)

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        out = self.model(x)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.conv_out(out)

        return out 


class SwinModel(nn.Module):
    def __init__(
        self,
        ch, in_channels, out_ch, swin_blk_num=4, swin_blk_depth=6,
    ): 
        super().__init__()

        self.code_emb = nn.Conv2d(in_channels, ch, 1, 1, 0, bias=False)
        self.combine = nn.Conv2d(ch * 2, ch, 1, 1, 0, bias=False)

        self.swin_layers = SwinLayers(embed_dim=ch, swin_dim=ch,
                                      blk_num=swin_blk_num, blk_depth=swin_blk_depth,
        )

        self.conv_out = nn.Sequential(
            normalize(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 1, 1, 0)
        )
    
    def forward(self, x):

        if isinstance(x, list):
            code_emb1 = self.code_emb(x[0])
            code_emb2 = self.code_emb(x[1])
            h = self.combine(torch.cat([code_emb1, code_emb2], dim=1)) 
        else:
            h = self.code_emb(x)

        h = self.swin_layers(h)
        out = self.conv_out(h)
                
        return out


class UNetModel(nn.Module):
    def __init__(
        self,
        ch, in_channels, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        resamp_with_conv=True, resolution=32, output_func=None, **ignore_kwargs
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.output_func = output_func

        # downsampling
        self.code_emb = nn.Conv2d(in_channels, ch, 1, 1, 0, bias=False)
        self.combine = nn.Conv2d(ch * 2, ch, 1, 1, 0, bias=False)

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # Encoder 
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.enc_blocks = nn.ModuleList()
        enc_feat_channels = []
        for i_level in range(self.num_resolutions):
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            enc_feat_channels.append(block_in) 
            tmp_block = []
            for i_block in range(self.num_res_blocks):
                tmp_block.append(ResBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         ))
                block_in = block_out
            if i_level != self.num_resolutions-1:
                tmp_block.append(Downsample(block_in))
            self.enc_blocks.append(nn.Sequential(*tmp_block))

        # middle
        self.mid_block = [ 
            ResBlock(in_channels=block_in, out_channels=block_in),
            AttnBlock(block_in),
        ] * 2 
        self.mid_block = nn.Sequential(*self.mid_block)

        # Decoder
        in_ch_mult = in_ch_mult[:-1]
        self.dec_blocks = nn.ModuleList()
        for level, mult in list(enumerate(ch_mult))[::-1]:
            block_in = ch * mult + enc_feat_channels.pop()
            block_out = ch * in_ch_mult[level]
            tmp_block = []
            for i in range(self.num_res_blocks):
                tmp_block.append(ResBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            if level != 0:
                tmp_block.append(Upsample(block_in))
            self.dec_blocks.append(nn.Sequential(*tmp_block))
    
        # end
        self.conv_out = nn.Sequential(
            normalize(block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, 1, 1, 0)
        )

    def forward(self, x):

        # h = self.conv_in(x)
        if isinstance(x, list):
            code_emb1 = self.code_emb(x[0])
            code_emb2 = self.code_emb(x[1])
            h = self.combine(torch.cat([code_emb1, code_emb2], dim=1)) 
        else:
            h = self.code_emb(x)

        enc_feats = []
        for m in self.enc_blocks:
            enc_feats.append(h)
            h = m(h)

        h = self.mid_block(h)

        for m in self.dec_blocks:
            h = torch.cat([h, enc_feats.pop()], dim=1)
            h = m(h)

        out = self.conv_out(h) 
        if self.output_func is not None:
            out = self.output_func(out)
        
        return out


    