RobRank: Adversarial Robustness in Deep Ranking
===

RobRank is extended from previous ECCV'2020 work [*"Adversarial Ranking
Attack and Defense,"*](https://github.com/cdluminate/advrank) with a major
code refactor. We newly introduce an "Anti-Collapse Triplet" defense method
which achieves at least 60% and at most 540% improvement in adversarial
robustness compared to the ECCV work.

Note, the project name is Rob**R**ank, instead of Rob**B**ank.

## 1. Common Usage of CLI

Python library `RobRank` provides these functionalities: (1) training 
classification or ranking (deep metric learning) models, either vanilla
or defensive; (2) perform adversarial attack against the trained models;
(3) perform batched adversarial attack. See below for detailed usage.

### 1.1. Training

Training deep metric learning model or classification model, either normally or adversarially.
As `pytorch-lightning` is used by this project, the training process will automatically use `DistributedDataParallel` when more than one GPU are available.

```shell
CUDA_VISIBLE_DEVICES=<GPUs> python3 train.py -C <dataset>:<model>:<loss>
```

where
* dataset (for all available datasets see robrank/datasets/__init__.py)
  * mnist, fashion, cub, cars, sop (for deep metric learning)
  * mnist, cifar10 (for classification)
* model (for all available models see robrank/models/__init__.py)
  * rres18: resnet 18 for deep metric learning (DML)
  * rres18d: resnet 18 for DML with EST defense
  * rres18p: resnet 18 for DML with ACT defense
  * csres18: resnet 18 for small-sized input (32x32)
* loss (for all available losses see robrank/losses/__init__.py)
  * ptripletN: triplet using Normalized Euclidean with SPC-2 batch.
  * ce: cross-entropy for classification

For example:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -C mnist:cc2f2:ce
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -C cifar10:csres18:ce
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -C mnist:rc2f2:ptripletN
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -C mnist:rc2f2p:ptripletN
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -C cub:rres18:ptripletN
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -C cub:rres18p:ptripletN
```

Tips:
1. export `FAISS_CPU=1` to disable NMI score calculation on GPU. This could
save a little bit of video memory.

### 1.2. Adversarial Attack

Script `advrank.py` is the entrance for conducting adversarial attacks
against a trained model. For example, to conduct CA (w=1) with several
manually specified PGD parameters, you can do it as follows:

```shell
python3 advrank.py -v -A CA:pm=+:W=1:eps=0.30196:alpha=0.011764:pgditer=32 -C <xxx.ckpt>
```
where `xxx.ckpt` is the path to the trained model (saved as a pytorch-lightning checkpoint).
The arguments specific to adversarial attacks are joined with a colon ":"
in order to avoid lengthy python code based `argparse` module.

Please browse the bash scripts under the `tools/` directory for examples
of other types of attacks discussed in the paper.

### 1.3. Batched Adversarial Attack

Script `swipe.py` is used for conducting a batch of attacks against a specified
model (pytorch-lightning checkpoint), automatically. And it will save the
output in json format as `<model_ckpt>.ckpt.<swipe_profile>.json`.
Available `swipe_profile` includes `rob28`, `rob224` for ERS score;
and `pami28`, `pami224` for CA and QA in various settings. A full list
of possible profiles can be found in `robrank/cmdline.py`. You can even
customize the code and create your own profile for batched evaluation.

```shell
python3 swipe.py -p rob28 -C logs_fashion-rc2f2-ptripletN/.../xxx.ckpt
```

Currently only single-GPU mode is supported.

## 2. Project Information

### 2.1. Directory Hierarchy

```
(the following directory tree is manually edited and annotated)
.
├── requirements.txt              Python deps (`pip install -r ...txt`)
├── advrank.py                    Entrance script for adversarial ranking.
├── swipe.py                      Entrance script for batched attack.
├── tfdump.py                     Entrance script for dumping tensorboard db.
├── train.py                      Entrance script for training models.
├── validate.py                   Entrance script for model validation.
├── robrank                       RobRank library.
│   ├── attacks                   Attack Implementations.
│   │   └── advrank*.py           Adversarial ranking attack (ECCV'2020).
│   ├── defenses/*                Defense Implementations.
│   ├── configs/*                 Configurations (incl. hyper-parameters).
│   ├── datasets/*                Dataset classes.
│   ├── models                    Models and base classes.
│   │   ├── template_classify.py  Base class for classification models.
│   │   ├── template_hybrid.py    Base class for Classification+DML models.
│   │   └── template_rank.py      Base class for DML/ranking models.
│   ├── losses/*                  Deep metric learning loss functions.
│   ├── cmdline.py                Command line interface implementation.
│   └── utils.py                  Miscellaneous utilities.
└── tools/*                       Miscellaneous tools for experiments.
```

### 2.2. Tested Platform

```
OS: Debian unstable (May 2021), Ubuntu LTS
Python: 3.8.5 (anaconda)
PyTorch: 1.7.1, 1.8.1
Python Dependencies: see requirements.txt
```

### 2.3. References and Bibtex

If you found the paper/code useful/inspiring, please consider citing my work:

```bibtex
@misc{robrank,
      title={Adversarial Attack and Defense in Deep Ranking}, 
      author={Mo Zhou and Le Wang and Zhenxing Niu and Qilin Zhang and Nanning Zheng and Gang Hua},
      year={2021},
      eprint={2106.03614},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Or if you need the Bibtex of the ECCV'20 version: [M. Zhou, et al. "Adversarial Ranking Attack and Defense," ECCV'2020.](https://github.com/cdluminate/advrank)

**Software Projects:**

1. https://github.com/Confusezius/Deep-Metric-Learning-Baselines
2. https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
3. https://github.com/idstcv/SoftTriple
4. https://github.com/KevinMusgrave/pytorch-metric-learning

### 2.4. Copyright and License

```
Copyright (C) 2019-2021, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
