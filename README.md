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
**Try-It-on-Colab:** [[`fashion:rc2f2p:ptripletN`]](https://colab.research.google.com/drive/1QC34RCadO0QCj-YUsLTUI9_pzqn8nrBH?usp=sharing)
[[`cars:rres18p:ptripletN`]](https://colab.research.google.com/drive/1jjDK4X64bIv7fLyMSlVs-btEMxxzgm6V?usp=sharing)

**News and Updates**

1. [2022-03-02] New paper based on this code base has been published: [Enhancing Adversarial Robustness for Deep Metric Learning, CVPR, 2022](https://github.com/cdluminate/robdml). Note, in this new paper, we further improved the benign performance, adversarial robustness, as well as training efficiency altogether for robust metric learning.

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
| MNIST | C2F2 | Triplet | HM  | 99.0 | 99.4 | 99.0 | 82.6 | 77.9 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| Fashion | C2F2 | Triplet | N/A | 87.6 | 92.7 | 84.9 | 77.8 | 4.5  |
| Fashion | C2F2 | Triplet | EST | 78.6 | 86.8 | 64.6 | 64.9 | 36.4 |
| Fashion | C2F2 | Triplet | ACT | 79.4 | 87.9 | 71.6 | 69.6 | 67.7 |
| Fashion | C2F2 | Triplet | HM  | 88.0 | 92.9 | 85.6 | 77.2 | 83.9 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| CUB | RN18 | Triplet | N/A | 53.9 | 66.4 | 26.1 | 59.5 | 3.8  |
| CUB | RN18 | Triplet | EST | 8.5  | 13.0 | 2.6  | 25.2 | 5.3  |
| CUB | RN18 | Triplet | ACT | 27.5 | 38.2 | 12.2 | 43.0 | 33.9 |
| CUB | RN18 | Triplet | HM  | 34.9 | 45.0 | 19.8 | 47.1 | 36.0 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| CARS | RN18 | Triplet | N/A | 62.5 | 74.0 | 23.8 | 57.0 | 3.6  |
| CARS | RN18 | Triplet | EST | 30.7 | 41.0 | 5.6  | 31.8 | 7.3  |
| CARS | RN18 | Triplet | ACT | 43.4 | 56.5 | 11.8 | 42.9 | 38.6 |
| CARS | RN18 | Triplet | HM  | 60.2 | 71.6 | 33.9 | 51.2 | 46.0 |

| Dataset | Model | Loss | Defense | R@1 | R@2 | mAP | NMI | ERS |
| ---     | ---   | ---  | ---     | --- | --- | --- | --- | --- |
| SOP | RN18 | Triplet | N/A | 62.9 | 68.5 | 39.2 | 87.4 | 4.0  |
| SOP | RN18 | Triplet | EST | 46.0 | 51.4 | 24.5 | 84.7 | 31.7 |
| SOP | RN18 | Triplet | ACT | 47.5 | 52.6 | 25.5 | 84.9 | 50.8 |
| SOP | RN18 | Triplet | HM  | 46.8 | 51.7 | 24.5 | 84.7 | 61.6 |

Source of these defense methods:

1. N/A: Just standard classification network.
2. EST: Adversarial Ranking Attack and Defense (ECCV2020)
3. ACT: Adversarial Attack and Defense in Deep Ranking (arXiv:2106.03614)
4. HM (or, concreately, ghmetsmi): Enhancing Adversarial Robustness for Deep Metric Learning (CVPR2022)

## 1. Common Usage of CLI

Python library `RobRank` provides these functionalities: (1) training 
classification or ranking (deep metric learning) models, either vanilla
or defensive; (2) perform adversarial attack against the trained models;
(3) perform batched adversarial attack. See below for detailed usage.

You can always specify the GPUs to use by `export CUDA_VISIBLE_DEVICES=<GPUs>`.

### 1.1. Training

Training deep metric learning model or classification model, either normally or
adversarially.  As `pytorch-lightning` is used by this project, the training
process will automatically use `DistributedDataParallel` when more than one GPU
are available.

The typical usage for training a model is as follows
```shell
python3 bin/train.py -C <dataset>:<model>:<loss>
```
where a "config" is composed of three components, so that such mechanism
is flexible enough to express many combinations. Specifically:

* `dataset` (for all available datasets see `robrank/datasets/__init__.py`)
  * mnist, fashion, cub, cars, sop (for deep metric learning)
  * mnist, cifar10 (for classification)
* model (for all available models see `robrank/models/__init__.py`)
  * cc2f2: c2f2 network for classification
  * cres18: resnet-18 for classification
  * rres18: resnet-18 for deep metric learning (DML)
  * rres18d: resnet-18 for DML with EST defense
  * rres18p: resnet-18 for DML with ACT defense
* loss (for all available losses see `robrank/losses/__init__.py`)
  * ce: cross-entropy for classification
  * ptripletN: triplet using Normalized Euclidean with SPC-2 batch.
  * ptripletE: triplet using Euclidean (not on unit hypersphere) with SPC-2 batch.
  * ptripletC: triplet using Cosine Distance with SPC-2 batch.
  * pmtripletN: ptripletN using semihard sampling instead of random
  * pstripletN: ptripletN using softhard sampling
  * pdtripletN: ptripletN using distance weithed sampling
  * phtripletN: ptripletN using batch hardest sampling

For example:
```shell
# classification
python3 bin/train.py -C mnist:cc2f2:ce --do_test
python3 bin/train.py -C cifar10:cres18:ce   # cifar10, resnet 18 classify, CE loss
python3 bin/train.py -C cifar10:cres50:ce   # cifar10, resnet 50 classify, CE loss
# deep metric learning
python3 bin/train.py -C mnist:rc2f2:ptripletN
python3 bin/train.py -C mnist:rc2f2p:ptripletN
python3 bin/train.py -C cub:rres18:ptripletN
python3 bin/train.py -C cub:rres18p:ptripletN
python3 bin/train.py -C cars:rres18:ptripletN
python3 bin/train.py -C cars:rres18p:ptripletN
python3 bin/train.py -C sop:rres18:ptripletN
python3 bin/train.py -C sop:rres18p:ptripletN
```

Tips:
1. When training DML models, export `FAISS_CPU=1` to disable NMI score
calculation on GPU (faiss). This could save a little bit of video memory of you
encounter CUDA OOM.
2. To change the number of PGD iterations for creating adversarial examples during
the training process, create an empty file to indicate the change. For example, 
`touch override_pgditer_8`. See `robrank/configs/configs_rank.py` for detail.

### 1.2. Adversarial Attack

Script `bin/advrank.py` is the entrance for conducting adversarial attacks
against a trained model. For example, to conduct CA (w=1) with several
manually specified PGD parameters, you can do it as follows:

```shell
python3 bin/advrank.py -v -A CA:pm=+:W=1:eps=0.30196:alpha=0.011764:pgditer=32 -C <xxx.ckpt>
```
where `xxx.ckpt` is the path to the trained model (saved as a pytorch-lightning checkpoint).
The arguments specific to adversarial attacks are joined with a colon ":"
in order to avoid lengthy python code based `argparse` module. Example:

```shell
python3 bin/advrank.py -v -A CA:pm=+:W=1:eps=0.30196:alpha=0.011764:pgditer=32 -C logs_cub-rres18p-ptripletN/lightning_logs/version_0/checkpoints/epoch=74-step=3974.ckpt
```

Please browse the bash scripts under the `tools/` directory for examples
of other types of attacks discussed in the paper. Example:

```shell
export CKPT=logs_cub-rres18p-ptripletN/lightning_logs/version_0/checkpoints/epoch=74-step=3974.ckpt
bash tools/ca.bash + $CKPT      # CA+ column
bash tools/ca.bash - $CKPT      # CA- column
bash tools/es.bash $CKPT        # ES:D and ES:R column
```

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
python3 bin/swipe.py -p rob224 -C logs_cub-rres18-ptripletN/.../xxx.ckpt
```

You may use `-m <number>` (e.g. `-m 10`) specify the max number of iterations
to get a quick accessment instead of going through the whole validation
dataset.

Currently only single-GPU mode is supported for attacks. When the batched
attack is finished, the results will be written into a json file
`logs_fashion-rc2f2-ptripletN/.../xxx.ckpt.json`.  A helper script
`tools/pjswipe.py` can display the content of resulting json files and
calculate the corresponding ERS:

```
$ python3 tools/pjswipe.py logs_fashion-rc2f2-ptripletN
```
The script will automatically use the json file corresponding to the latest
version of the specified config. So specifying the log directory is enough.
That said, if multiple versions of the same config exists, and you want to
let it print result of an old version, export `ITH=<version>` (e.g. `ITH=1`)
and run again. If tested with multiple profiles, export `JTYPE` to select
exact profile. Read the comments in `tools/pjswipe.py` for details.

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

Tested Software versions:

```
OS: Debian unstable, Debian Bullseye, Ubuntu 20.04 LTS, Ubuntu 16.04 LTS
Python (anaconda distribution): 3.8.5, 3.9.X
PyTorch: 1.7.1, 1.8.1, 1.11.0
PyTorch-Lightning: see requirements.txt
```

Mainly Tested Hardware:
```
CPU: Intel Xeon Family
GPU: Nvidia GTX1080Ti, Titan Xp, RTX3090, A5000, A6000, A100
```
With 8 RTX3090 GPUs, most experiments can be finished within 1 day.
With older configurations (such as `4* GTX1080Ti`), most experiments can be
finished within 3 days, including adversarial training.

Memory requirement: 12GB video memory is required for adversarial training of
RN18, Mnas, and IBN. Additionally, adversarial training of RN50 requires 24GB.

If you encounter the following error message:
```
Traceback (most recent call last):
  File "bin/train.py", line 16, in <module>
    import robrank as rr
ModuleNotFoundError: No module named 'robrank'
```
Just try `export PYTHONPATH=.` and run your command again.

### 2.3. Dataset Preparation

The default data path setting for any dataset can be found in
[`robrank/configs/configs_dataset.py`](robrank/configs/configs_dataset.py).

**MNIST** and **Fashion-MNIST** are downloaded using torchvision. The helper script
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

**CUB:** The tarball can be downloaded from `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`. Then change your working directory to `~/.torch` and `tar xvf <path>/CUB_200_2011.tgz -C .`. Now we are all set.

**CARS:** Create a directory `~/.torch/cars` then change working directory into it. Download `http://imagenet.stanford.edu/internal/car196/car_ims.tgz`
and `http://imagenet.stanford.edu/internal/car196/cars_annos.mat` into the directory. In the end extract the tarball `tar xvf car_ims.tgz`. We are ready to go.

**SOP:** After you downloaded `Stanford_Online_Products.zip` from `ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip`,
just do `$ cd ~/.torch` and `$ unzip <path>/Stanford_Online_Products.zip`. Now SOP is ready to use.

The dataset loader is able to smartly read the dataset from `/dev/shm` to
overcome IO bottleneck (especially from HDDs) if a copy of dataset if available
there. For instance, `rsync -av ~/.torch/Stanford_Online_Products /dev/shm`.

**CIFAR:** For cifar10 `cd ~/.torch/; wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz; tar xvf cifar-10-python.tar.gz`. And for cifar100 `cd ~/.torch/; wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz; tar xvf cifar-100-python.tar.gz`.

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
8. https://github.com/MadryLab/robustness

## Frequently Asked Questions

* Q: Concrete code position of the defense methods?

A: As you may have find it ... there are lots of leftover attemps towards a
better defense in `robrank/defenses`. And renames during research process
also results in some inconsistency. So I'd better directly point out the
code position here:  
(1) `hm_training_step` in [`defenses/amd.py`](https://github.com/cdluminate/robrank/blob/main/robrank/defenses/amd.py)
is the Hardness Manipulation (HM) defense. The function for creating adversarial
examples for adversarial training is `MadryInnerMax.HardnessManipulate` in the same file.  
(2) `pnp_training_step` in [`defenses/pnp.py`](https://github.com/cdluminate/robrank/blob/main/robrank/defenses/pnp.py)
is the Anti-Collapse Triplet (ACT) defense. The function for creating adversarial examples for adversarial
training is `PositiveNegativePerplexing.pncollapse` in the same file.  
(3) `est_training_step` in [`defenses/est.py`](https://github.com/cdluminate/robrank/blob/main/robrank/defenses/est.py)
is the Embedding-Shift Triplet (EST) defense. The function for creating adversarial examples for adversarial
training is the ES attack from the [`AdvRank` class](https://github.com/cdluminate/robrank/blob/main/robrank/attacks/advrank.py).  

* Q: Training stuck at the end of validation with Nvidia A100, A6000, A5000, RTX3090, etc.

A: I hate Nvidia for such weird issue. And the reason of distributed data parallel
being stuck varies across different situations or machines.
Here are a bunch of tricks that might or might not work:  
(1) Comment out `th.distributed.barrier()` from the code and run again.
You can locate that barrier function in the code using ripgrep. This seemed effective on RTX3090;  
(2) use [`rank_zero_only` option](https://github.com/PyTorchLightning/pytorch-lightning/issues/8821#issuecomment-902402784) for pytorch-lightning logger:
`sed -i robrank/models/template_rank.py -e "s/self.log(\(.*\))/self.log(\1, rank_zero_only=True)/g"`;  
(3) [change the distributed backend](https://github.com/PyTorchLightning/pytorch-lightning/discussions/6509) of [pytorch](https://pytorch.org/docs/stable/distributed.html#debugging-torch-distributed-applications): `export PL_TORCH_DISTRIBUTED_BACKEND=gloo`;  
(4) disable P2P feature for NCCL. `export NCCL_P2P_DISABLE=1`;  
(5) change accelerator from `ddp` to `ddp_spawn` in `robrank/cmdline.py`. Run the training again and let it raise error.
Then change back to `ddp` and the A5000 started working;  
(6) [P2P GPU traffic will fail with IOMMU](https://github.com/pytorch/pytorch/issues/1637#issuecomment-338268158). Check the `p2pBandwithLatencyTest` cuda example and see whether it could run. If not, then it's not a pytorch issue. Disable `iommu` from kernel parameter should work. `GRUB_CMDLINE_LINUX="iommu=soft"` in `/etc/default/grub`. Run `sudo update-grub2` after edit. Linux kernel has a documentation describing [this iommu parameter](https://www.kernel.org/doc/Documentation/x86/x86_64/boot-options.txt). IOMMU group assignment can be found under `/sys/kernel/iommu_group`;  
(7) Use only even/odd numbered GPUs `CUDA_VISIBLE_DEVICES=1,3,5` instead of `CUDA_VISIBLE_DEVICES=1,2,3`. This works sometimes for at least the `p2pBandwithLatencyTest` test program;  
(8) turn off ACS in BIOS;  
(9) change `num_workers=0` for dataloader.  

* Q: Maxepoch is 16 or 150 in the paper, but 8 or 75 in the code?

A: They are equivalent due to the implementation details in the dataset
sampler. It is a fixable problem (but not necessary). See [issue #9](https://github.com/cdluminate/robrank/issues/9).

* Q: Training time?

RTX A5000 performance is similar to RTX 3090. RTX A6000 is slightly faster
than RTX 3090. Nvidia A100 is roughly 1.5 times faster than RTX 3090.
RTX 3090 is roughly 2~3 times faster than Nvidia Titan Xp (or GTX 1080Ti).
In the following table, `eta` is exactly PGD iteration number (pgditer).
It can be overriden by file indicators like `override_pgditer_8` as described
in previous documentation. Time cost on MNIST and Fashion-MNIST is expected
to be identical. For the rest datasets, time consumption order is CUB < CARS < SOP.

| Config                              | eta | GPU Model | Number of GPUs | Time (roughly) |
| ---                                 | --- | ---       | ---            | ---            |
| `fashion:rc2f2:ptripletN`           | N/A | Titan Xp  | 2 (DDP)        | 2 min          |
| `fashion:rc2f2p:ptripletN`          | 32  | Titan Xp  | 2 (DDP)        | 10 min         |
| `fashion:rc2f2ghmetsmi:ptripletN`   | 32  | RTX A5000 | 4 (DDP)        | 8 min          |
| `cub:rres18:ptripletN`              | N/A | Titan Xp  | 2 (DDP)        | 30 min         |
| `cub:rres18:ptripletN`              | N/A | RTX A5000 | 4 (DDP)        | 10 min         |
` `cub:rres18p:ptripletN`             | 8   | Titan Xp  | 2 (DDP)        |                |
| `cub:rres18p:ptripletN`             | 32  | Titan Xp  | 2 (DDP)        | 420 min        |
| `cub:rres18p:ptripletN`             | 32  | RTX A5000 | 4 (DDP)        | 120 min        |
| `cub:rres18p:ptripletN`             | 32  | RTX A6000 | 2 (DDP)        | 180 min        |
| `cub:rres18ghmetsm:ptripletN`       | 32  | RTX A5000 | 4 (DDP)        | 120 min        |
| `cub:rres18ghmetsmi:ptripletN`      | 32  | Titan Xp  | 2 (DDP)        | 470 min        |
| `cub:rres18ghmetsmi:ptripletN`      | 32  | RTX A5000 | 4 (DDP)        | 120 min        |
| `cars:rres18:ptripletN`             | N/A | RTX A5000 | 4 (DDP)        | 15 min         |
` `cars:rres18p:ptripletN`            | 8   | Titan Xp  | 2 (DDP)        |                |
| `cars:rres18p:ptripletN`            | 32  | RTX A5000 | 4 (DDP)        | 150 min        |
| `cars:rres18ghmetsmi:ptripletN`     | 32  | Titan Xp  | 2 (DDP)        | 530 min        |
| `sop:rres18:ptripletN`              | N/A | RTX A5000 | 4 (DDP)        | 60 min         |
| `sop:rres18p:ptripletN`             | 32  | RTX A6000 | 2 (DDP)        |                |

* Q: Pretrained Weights?

Indeed that will be very useful for comparison.
This is under preparation.

| Config                                 | Download |
| ---                                    | ---      |
| `cub:rres18p:ptripletN`                | TODO     |
| `cub:rres18ghmetsmi:ptripletN`         | TODO     |
| `cars:rres18p:ptripletN`               | TODO     |
| `cars:rres18ghmetsmi:ptripletN`        | TODO     |
| `sop:rres18p:ptripletN`                | TODO     |
| `sop:rres18ghmetsmi:ptripletN`         | TODO     |

### Copyright and License

```
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

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
