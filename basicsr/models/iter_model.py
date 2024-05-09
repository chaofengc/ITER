from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa
import gc


@MODEL_REGISTRY.register()
class ITERModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # define metric functions 
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items(): 
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook 
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            hq_opt = self.opt['network_g'].copy()
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
    
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # define losses
        if train_opt.get('struct_opt'):
            self.cri_struct = pyiqa.create_metric(train_opt['struct_opt']['type'], as_loss=True) 
        else:
            self.cri_struct = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.lq = F.interpolate(self.lq, (64, 64))
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            # self.gt = F.interpolate(self.gt, (256, 256))
        
        if 'text' in data.keys():
            self.text = data['text']
        else:
            self.text = None
    
    def set_acc(self, acc):
        self.accelerator = acc
        self.acc_list = self.accelerator.prepare(self.net_g, self.net_d, self.net_hq, self.cri_perceptual, self.optimizer_g, self.optimizer_d)

    def optimize_parameters(self, current_iter):
        net_g, net_d, net_hq, cri_pcp, optimizer_g, optimizer_d = self.acc_list

        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        optimizer_g.zero_grad()

        if self.LQ_stage:
            with torch.no_grad():
                self.gt_rec, _, _, gt_indices, _ = net_hq(self.gt)

            self.output, l_codebook, self.refined_output, _, self.ret_masks = net_g(self.lq, gt_indices, text=self.text) 
        else:
            self.output, l_codebook, self.refined_output, _, self.ret_masks = self.net_g(self.gt, text=self.text) 

        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook
            loss_dict['l_codebook'] = l_codebook

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix 

            # l_pix2 = self.cri_pix(self.refined_output, self.gt)
            # l_g_total += l_pix2 

            loss_dict['l_pix'] = l_pix
        
        if self.cri_struct:
            l_pix = 0
            l_pix += (1 - self.cri_struct(self.output, self.gt).mean())
            l_pix = l_pix * train_opt['struct_opt']['loss_weight'] 
            l_g_total += l_pix
            loss_dict['l_struct'] = l_pix
        
        # perceptual loss
        if self.cri_perceptual:

            # l_percep, l_style = cri_pcp(self.output, self.gt)
            l_percep, l_style = cri_pcp(torch.cat([self.output, self.refined_output], dim=0), torch.cat([self.gt, self.gt], dim=0))
            # pixel level loss for demask network
            # l_percep2, l_style2 = cri_pcp(self.refined_output, self.gt)
            # l_percep += l_percep2 

            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style

        pred_for_d = self.output 
        gt_for_d = self.gt 
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = net_d(pred_for_d)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # l_g_total.mean().backward()
        # self.optimizer_g.step()
        self.accelerator.backward(l_g_total)
        optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in net_d.parameters():
                p.requires_grad = True
            optimizer_d.zero_grad()
            # real
            real_d_pred = net_d(gt_for_d.float())
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            # l_d_real.backward()
            self.accelerator.backward(l_d_real)
            # fake
            fake_d_pred = net_d(pred_for_d.float().detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            # l_d_fake.backward()
            self.accelerator.backward(l_d_fake)
            optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad() 
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape
        # if h*w < min_size:
        #     self.output = net_g.test(lq_input)
        # else:
        #     self.output = net_g.test_tile(lq_input)
        self.output, _ = net_g.test(lq_input)
        self.net_g.train()
        return self.output
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            
        pbar = tqdm(total=len(dataloader), unit='image')

        if hasattr(self, 'x_crop'): 
            del self.x_crop
            del self.y_crop
        del self.lq
        del self.output
        torch.cuda.empty_cache()
        gc.collect()

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric') 
        
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            gc.collect()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.jpg')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.jpg')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.jpg')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item() 
            
            del self.gt
            del sr_img 
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '', except_key='vqgan')
                    self.save_network(self.net_d, 'net_d_best', '', except_key='vqgan')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '', except_key='vqgan')
                    self.save_network(self.net_d, 'net_d_best', '', except_key='vqgan')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    @torch.no_grad()
    def get_current_visuals(self):
        vis_samples = 8 
        out_dict = OrderedDict()
        out_dict['lq'] = torch.nn.functional.interpolate(self.lq, self.output.shape[2:], mode='bicubic', align_corners=False).detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]

        if hasattr(self, 'x_crop'):
            out_dict['patch_sample'] = torch.cat([self.x_crop, self.y_crop], dim=2).detach().cpu()[:vis_samples]

        if self.refined_output is not None:
            out_dict['result'] = torch.cat([out_dict['result'], self.refined_output.detach().cpu()[:vis_samples]], dim=2)

        if self.ret_masks is not None:
            vis_masks = torch.nn.functional.interpolate(torch.cat(self.ret_masks, dim=2).float(), scale_factor=2)
            out_dict['masks_train'] = vis_masks.detach().cpu()[:vis_samples] 

            # get refined result in eval mode
            self.net_g.eval()
            net_g = self.get_bare_model(self.net_g)
            output, _, _, _, ret_masks = net_g(self.lq[:vis_samples]) 
            output = torch.cat(output, dim=2)
            out_dict['result_refined'] = output[:vis_samples]
            vis_masks = torch.nn.functional.interpolate(torch.cat(ret_masks, dim=2).float(), scale_factor=2)
            out_dict['masks_eval'] = vis_masks.detach().cpu()[:vis_samples] 
            self.net_g.train()

        vis_codebook = self.opt['logger'].get('vis_codebook', False)
        if not self.LQ_stage and vis_codebook:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'smoothed_lq'):
            out_dict['smoothed_lq'] = self.smoothed_lq.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        
        for k, v in out_dict.items():
            out_dict[k] = v.float().clamp(0, 1)

        return out_dict

    def save(self, epoch, current_iter):
        if self.accelerator.is_main_process:
            self.save_network(self.net_g, 'net_g', current_iter, except_key='vqgan')
            self.save_network(self.net_d, 'net_d', current_iter, except_key='vqgan')
            self.save_training_state(epoch, current_iter)
