'''
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
'''

from dataclasses import dataclass

#########################
# Loss Function Configs #
#########################


@dataclass
class contrastive:
    margin_cosine: float = 1.0
    margin_euclidean: float = 1.0


@dataclass
class triplet:
    '''
    Setting margin_cosine: float = 0.8 can further improve the model
    robustness with EST or ACT defense. But here we don't enable that
    by default.
    '''
    margin_cosine: float = 0.2
    margin_euclidean: float = 1.0


@dataclass
class quadruplet:
    margin2_cosine: float = 0.1
    margin2_euclidean: float = 0.5


@dataclass
class glift:
    margin_cosine: float = 0.2
    margin_euclidean: float = 1.0
    l2_weight: float = 5e-3


@dataclass
class npair:
    l2_weight: float = 5e-3


@dataclass
class margin:
    beta: float = 0.6  # [locked, ICML20]
    lr_beta: float = 5e-4  # [locked, ICML20]
    margin: float = 0.2


@dataclass
class multisim:
    pos_weight: float = 2  # [locked, ICML20]
    neg_weight: float = 40  # [locked]
    margin: float = 0.1  # [locked]
    threshold: float = 0.5  # [locked]


@dataclass
class snr:
    margin: float = 0.2  # [locked, ICML20]
    reg_lambda: float = 0.005  # [locked, ICML20]
