RobRank: Adversarial Robustness in Deep Ranking
===

![badage](https://github.com/cdluminate/robrank/actions/workflows/github-actions-demo.yml/badge.svg)
[![GitHub license](https://img.shields.io/github/license/cdluminate/robrank)](https://github.com/cdluminate/robrank/blob/main/LICENSE)

Deep neural networks are vulnerable to adversarial attacks, and so does deep
ranking or deep metric learning models. The project *RobRank* aims to study
the empirical adversarial robustness of deep ranking / metric learning models.
Our contribution includes (1) the definition and implementation of two new
adversarial attacks, namely candidate attack and query attack; (2) two
adversarial defense methods (based on adversarial training) are proposed
to improve model robustness against a wide range of attacks; (3) a comprehensive
empirical robustness score for quantitatively assessing adversarial robustness.
In particular, an **"Anti-Collapse Triplet" defense** method is newly introduced
in *RobRank*, which **achieves at least 60% and at most 540% improvement in
adversarial robustness** compared to the ECCV work. See the preprint manuscript
for details.

RobRank codebase is extended from my previous ECCV'2020 work [*"Adversarial
Ranking Attack and Defense,"*](https://github.com/cdluminate/advrank) with
a major code refactor. You may find most functionalities of the previous
codebase in this repository as well.

Note, the project name is Rob**R**ank, instead of Rob**B**ank.

**Preprint-Title:** "Adversarial Attack and Defense in Deep Ranking"  
**Preprint-Authors:** Mo Zhou, Le Wang, Zhenxing Niu, Qilin Zhang, Nanning Zheng, Gang Hua  
**Preprint-Link:** https://arxiv.org/abs/2106.03614  
**Keywords:** Deep {Ranking, Metric Learning}, Adversarial {Attack, Defense, Robustness}  

**Project Status:** Actively maintained.  
**Install-RobRank-Python-Dependency:** `$ pip install -r requirements.txt`  
**Install-RobRank:** `$ python3 setup.py install` (this is optional)  

**News and Updates**

...

## Tables for Robustness Comparison

In the following tables, "N/A" denotes "no defense equipped"; EST is the
defense proposed in the ECCV'2020 paper; ACT is the new defense in the preprint
paper. These rows are sorted by ERS. I'm willing to add other DML defenses for
comparison in these tables.

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| MNIST | C2F2 | Triplet | N/A | 99.0 | 99.4 | 98.7 | 84.7 | 13.3 |
| MNIST | C2F2 | Triplet | EST | 98.3 | 99.0 | 91.3 | 80.7 | 40.5 |
| MNIST | C2F2 | Triplet | ACT | 98.6 | 99.1 | 98.1 | 86.4 | 78.6 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| Fashion | C2F2 | Triplet | N/A | 87.6 | 92.7 | 84.9 | 77.8 | 4.5  |
| Fashion | C2F2 | Triplet | EST | 78.6 | 86.8 | 64.6 | 64.9 | 36.4 |
| Fashion | C2F2 | Triplet | ACT | 79.4 | 87.9 | 71.6 | 69.6 | 67.7 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| CUB | RN18 | Triplet | N/A | 53.9 | 66.4 | 26.1 | 59.5 | 3.8  |
| CUB | RN18 | Triplet | EST | 8.5  | 13.0 | 2.6  | 25.2 | 5.3  |
| CUB | RN18 | Triplet | ACT | 27.5 | 38.2 | 12.2 | 43.0 | 33.9 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| CARS | RN18 | Triplet | N/A | 62.5 | 74.0 | 23.8 | 57.0 | 3.6  |
| CARS | RN18 | Triplet | EST | 30.7 | 41.0 | 5.6  | 31.8 | 7.3  |
| CARS | RN18 | Triplet | ACT | 43.4 | 56.5 | 11.8 | 42.9 | 38.6 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| SOP | RN18 | Triplet | N/A | 62.9 | 68.5 | 39.2 | 87.4 | 4.0  |
| SOP | RN18 | Triplet | EST | 46.0 | 51.4 | 24.5 | 84.7 | 31.7 |
| SOP | RN18 | Triplet | ACT | 47.5 | 52.6 | 25.5 | 84.9 | 50.8 |

## 1. Common Usage of CLI

Python library `RobRank` provides these functionalities: (1) training 
classification or ranking (deep metric learning) models, either vanilla
or defensive; (2) perform adversarial attack against the trained models;
(3) perform batched adversarial attack. See below for detailed usage.

You can always specify the GPUs to use by `export CUDA_VISIBLE_DEVICES=<GPUs>`.

### 1.1. Training

Training deep metric learning model or classification model, either normally or adversarially.
As `pytorch-lightning` is used by this project, the training process will automatically use `DistributedDataParallel` when more than one GPU are available.

```shell
python3 bin/train.py -C <dataset>:<model>:<loss>
```

where
* dataset (for all available datasets see `robrank/datasets/__init__.py`)
  * mnist, fashion, cub, cars, sop (for deep metric learning)
  * mnist, cifar10 (for classification)
* model (for all available models see `robrank/models/__init__.py`)
  * rres18: resnet 18 for deep metric learning (DML)
  * rres18d: resnet 18 for DML with EST defense
  * rres18p: resnet 18 for DML with ACT defense
* loss (for all available losses see `robrank/losses/__init__.py`)
  * ptripletN: triplet using Normalized Euclidean with SPC-2 batch.
  * ce: cross-entropy for classification

For example:
```shell
python3 bin/train.py -C mnist:cc2f2:ce --do_test
python3 bin/train.py -C mnist:rc2f2:ptripletN
python3 bin/train.py -C mnist:rc2f2p:ptripletN
python3 bin/train.py -C cub:rres18:ptripletN
python3 bin/train.py -C cub:rres18p:ptripletN
```

Tips:
1. export `FAISS_CPU=1` to disable NMI score calculation on GPU. This could
save a little bit of video memory of you encounter CUDA OOM.

### 1.2. Adversarial Attack

Script `bin/advrank.py` is the entrance for conducting adversarial attacks
against a trained model. For example, to conduct CA (w=1) with several
manually specified PGD parameters, you can do it as follows:

```shell
python3 bin/advrank.py -v -A CA:pm=+:W=1:eps=0.30196:alpha=0.011764:pgditer=32 -C <xxx.ckpt>
```
where `xxx.ckpt` is the path to the trained model (saved as a pytorch-lightning checkpoint).
The arguments specific to adversarial attacks are joined with a colon ":"
in order to avoid lengthy python code based `argparse` module.

Please browse the bash scripts under the `tools/` directory for examples
of other types of attacks discussed in the paper.

### 1.3. Batched Adversarial Attack

Script `bin/swipe.py` is used for conducting a batch of attacks against a specified
model (pytorch-lightning checkpoint), automatically. And it will save the
output in json format as `<model_ckpt>.ckpt.<swipe_profile>.json`.
Available `swipe_profile` includes `rob28`, `rob224` for ERS score;
and `pami28`, `pami224` for CA and QA in various settings. A full list
of possible profiles can be found in `robrank/cmdline.py`. You can even
customize the code and create your own profile for batched evaluation.

```shell
python3 bin/swipe.py -p rob28 -C logs_fashion-rc2f2-ptripletN/.../xxx.ckpt
```

Currently only single-GPU mode is supported. When the batched attack is finished,
the script `tools/pjswipe.py` can display the content of resulting
json files and calculate the corresponding ERS.

## 2. Project Information

### 2.1. Directory Hierarchy

```
(the following directory tree is manually edited and annotated)
.
├── requirements.txt              Python deps (`pip install -r ...txt`)
├── bin/train.py                  Entrance script for training models.
├── bin/advrank.py                Entrance script for adversarial ranking.
├── bin/swipe.py                  Entrance script for batched attack.
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

Tested Software:

```
OS: Debian unstable (May 2021), Ubuntu LTS
Python: 3.8.5 (anaconda)
PyTorch: 1.7.1, 1.8.1
```

Mainly Tested Hardware:
```
CPU: Intel Xeon 6226R
GPU: Nvidia RTX3090 (24GB)
```
With 8 RTX3090 GPUs, most experiments can be finished within 1 day.
With older configurations (such as `4* GTX1080Ti`), most experiments can be
finished within 3 days, including adversarial training.

Memory requirement: 12GB video memory is required for adversarial training of
RN18, Mnas, and IBN. Additionally, adversarial training of RN50 requires 24GB.

### 2.3. Dataset Preparation

The default data path setting for any dataset can be found in
[`robrank/configs/configs_dataset.py`](robrank/configs/configs_dataset.py).

MNIST and Fashion-MNIST are downloaded using torchvision. The helper script
`bin/download.py` can download and extract the two datasets for you.
Just do as follows in your terminal from the root directory of this project:
```shell
$ export PYTHONPATH=.
$ pyhton3 bin/download.py
```
Then the MNIST and Fashion-MNIST datasets are ready to use. Try to train a model.

The rest datasets, namely
[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),
[Cars-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), and
[Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
can be downloaded from their correspoding websites (and then manually
extracted). 

After you downloaded [`Stanford_Online_Products.zip`](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip),
just do `$ cd ~/.torch` and `$ unzip <path>/Stanford_Online_Products.zip`.

### 2.4. References and Bibtex

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

Bibtex of [M. Zhou, et al. "Adversarial Ranking Attack and Defense," ECCV'2020.](https://github.com/cdluminate/advrank) can be found in the linked page.

**Reference Software Projects:**

1. https://github.com/Confusezius/Deep-Metric-Learning-Baselines
2. https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
3. https://github.com/idstcv/SoftTriple
4. https://github.com/KevinMusgrave/pytorch-metric-learning
5. https://github.com/RobustBench/robustbench
6. https://github.com/fra31/auto-attack
7. https://github.com/KevinMusgrave/powerful-benchmarker

### Copyright and License

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
