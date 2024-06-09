import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
import gc
import numpy as np

from basicsr.utils import img2tensor, tensor2img, imwrite, set_random_seed 
from basicsr.utils.download_util import load_file_from_url 

from basicsr.archs.iter_arch import ITER 
from pyiqa import create_metric


pretrain_model_url = {
    'x2': 'https://github.com/chaofengc/ITER/releases/download/v0.1.0/ITER_x2.pth',
    'x4': 'https://github.com/chaofengc/ITER/releases/download/v0.1.0/ITER_x4.pth',
    'swinvqgan': 'https://github.com/chaofengc/ITER/releases/download/v0.1.0/ITER_swinvqgan.pth',
}


def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-wh', '--weight_hq', type=str, default=None, help='path for autoencoder model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('-si', '--save_intermediate', action='store_true', help='Save intermediate results')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-gt', '--gt_path', type=str, default=None, help='Directory of ground truth images')
    parser.add_argument('--max_size', type=int, default=720, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # swin vqgan model options 
    if args.weight_hq is None:
        swinvqgan_path = load_file_from_url(pretrain_model_url['swinvqgan'])
    else:
        swinvqgan_path = args.weight_hq

    model_opts = {
        'scale_factor': args.out_scale, 
        'LQ_stage': True, 
        'codebook_params': [[32, 1024, 512]],
        'vqgan_opt': {
            'ckpt_path': swinvqgan_path,
            'n_embed': 512,
            'embed_dim': 128,
            'ddconfig': {
                'in_channels': 3,
                'emb_dim': 128,
                'resolution': [256, 256],
                'nf': 128,
                'swin_dim': 256,
                'swin_window': 8,
                'swin_blk_depth': 6,
                'swin_blk_num': 1,
                'ch_mult': [1, 2, 2, 4],
            }
        }
    }

    if args.weight is None:
        weight_path = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    else:
        weight_path = args.weight
    print(f'Loading weight from {weight_path}')
    
    # set up the model
    sr_model = ITER(**model_opts).to(device, dtype=torch.bfloat16)
    sr_model.load_state_dict(torch.load(weight_path, map_location='cpu')['params'], strict=False)
    sr_model.eval()
    sr_model.set_sample_params(*[8, 1.0, 1.0, 0.5, 'linear'])
    set_random_seed(123)

    args.output = f'{args.output}_x{args.out_scale}'
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    
    metric_fns = [create_metric('niqe'), create_metric('pi')]
    metrics = 'NIQE, PI'
    if args.gt_path:
        gt_paths = sorted(glob.glob(os.path.join(args.gt_path, '*')))
        metric_fns.append(create_metric('lpips'))    
        metrics += ', LPIPS'

    np.set_printoptions(precision=4)
    scores = []
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = args.max_size ** 2 
        h, w = img_tensor.shape[2:]

        with torch.cuda.amp.autocast():
            if h * w < max_size:
                if args.save_intermediate:
                    output, inter_results = sr_model.test(img_tensor.half(), return_inter_results=args.save_intermediate)
                else:
                    output, inter_results = sr_model.test(img_tensor.half(), return_inter_results=args.save_intermediate)
            else:
                if args.save_intermediate:
                    print(f'Warning: save_intermediate is not supported for tiled_test {path}, size(hxw): {h}, {w}')
                output = sr_model.test_tile(img_tensor.half())
                inter_results = None

        img_name = img_name.replace('.png', '.jpg')
        output_img = tensor2img(output)
        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)

        # calculate metrics
        if args.gt_path:
            scores.append([fn(save_path, gt_paths[idx]).item() for fn in metric_fns])
        else:
            scores.append([fn(save_path).item() for fn in metric_fns])
        avg_scores = np.array(scores).mean(axis=0)

        if inter_results is not None:
            inter_results = tensor2img(inter_results)
            save_path = os.path.join(args.output, f'inter_results_{img_name}')
            imwrite(inter_results, save_path)
        
        pbar.set_description(f'Test {img_name}. Average [{metrics}] {avg_scores}')
        pbar.update(1)

        # free cache
        del img_tensor 
        del output 
        torch.cuda.empty_cache()
        gc.collect()

    pbar.close()


if __name__ == '__main__':
    main()