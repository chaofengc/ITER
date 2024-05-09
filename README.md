<div align="center">

## [Iterative Token Evaluation and Refinement for Real-World Super-Resolution](https://arxiv.org/abs/2312.05616)

[<sup>1</sup>Chaofeng Chen](https://chaofengc.github.io), [<sup>1</sup>Shangchen Zhou](https://shangchenzhou.com/), [<sup>1</sup>Liang Liao](https://liaoliang92.github.io/homepage/), [<sup>1</sup>Haoning Wu](https://teowu.github.io/), [<sup>2</sup>Wenxiu Sun](https://scholar.google.com/citations?user=X9lE6O4AAAAJ&hl=en), [<sup>2</sup>Qiong Yan](https://scholar.google.com/citations?user=uT9CtPYAAAAJ&hl=en), [<sup>1</sup>Weisi Lin](https://personal.ntu.edu.sg/wslin/Home.html)  
<sup>1</sup>S-Lab, Nanyang Technological University, <sup>2</sup>Sensetime Research

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.05616) ![arXiv](https://img.shields.io/badge/AAAI-2024-red.svg) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=chaofengc/ITER)

![teaser_img](./assets/fig_teaser.jpg)

</div>

-----------------------------

![framework_img](assets/fig_framework.jpg)

**Pipeline of ITER.** The input $I_l$ first passes through a distortion removal network $E_l$ to obtain the initially restored tokens $S_l$, which are composed of indexes of the quantized features in the codebook of VQGAN. Then, a reverse discrete diffusion process, conditioned on $S_l$, is used to generate textures. The process starts from completely masked tokens $S_T$. The refinement network (also called the de-masking network) $\phi_r$ generates refined outputs $S_{T-1}$ with $S_l$ as a condition. Then, $\phi_e$ evaluates $S_{T-1}$ to obtain the evaluation mask $m_{T-1}$, which determines the tokens to keep and refine for step $T-1$ through a masked sampling process. Repeat this process $T$ times to obtain de-masked outputs $S_0$, and then reconstruct the restored images $I_{sr}$ using the VQGAN decoder $D_H$. We found that $T\leq8$ is enough to get good results with ITER, which is much more efficient than other diffusion-based approaches.

## TODO List

- [ ] Release `ITER_x4` model.
- [x] Release `ITER_x2` model.
- [x] Release training and testing codes.
- [x] Release training datasets.

## 🔧 Dependencies and Installation

```
# git clone this repository
git clone https://github.com/chaofengc/ITER.git
cd ITER 

# create new anaconda env
conda create -n iter python=3.8
source activate iter 

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## ⚡Quick Inference

```
python inference_iter.py -s 2 -i ./testset/lrx4/frog.png
python inference_iter.py -s 4 -i ./testset/lrx4/frog.png
```

## 👨‍💻Train the model

### ⏬ Download Datasets

The training datasets can be downloaded from [🤗hugging face](https://huggingface.co/datasets/chaofengc/ITER). You may also refer to [FeMaSR](https://github.com/chaofengc/FeMaSR) to prepare your own training data. 

### ‍🔁 Training

Below are brief examples for training the model. **Please modify the corresponding configuration files to suit your needs.**

#### Stage I: Train the Swin-VQGAN

```
accelerate launch --multi_gpu --num_processes=8 --mixed_precision=bf16 basicsr/train.py -opt options/train_ITER_HQ_stage.yml
```

#### Stage II & III: Train the LQ encoder and the refinement network

``` 
accelerate launch --main_process_port=29600 --multi_gpu --num_processes=8 --mixed_precision=bf16 basicsr/train.py -opt options/train_ITER_LQ_stage_X2.yml

accelerate launch --main_process_port=29600 --multi_gpu --num_processes=8 --mixed_precision=bf16 basicsr/train.py -opt options/train_ITER_LQ_stage_X4.yml
```

## 📝 Citation

If you find this code useful for your research, please cite our paper:
```
@inproceedings{chen2024iter,
  title={Iterative Token Evaluation and Refinement for Real-World Super-Resolution},
  author={Chaofeng Chen and Shangchen Zhou and Liang Liao and Haoning Wu and Wenxiu Sun and Qiong Yan and Weisi Lin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024},
}
```

## ⚖️ License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a> and [NTU S-Lab License 1.0](./LICENCE_S-Lab).

## ❤️ Acknowledgement

This project is based on [BasicSR](https://github.com/xinntao/BasicSR).