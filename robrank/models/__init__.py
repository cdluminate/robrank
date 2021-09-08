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

# Naming conventions for these models
# <task><backbone><modifier>
# task in {c, r, h} for classification, ranking, hybrid
# backbone e.g. lenet, res18, mnas10
# moddifier in {, d, e} for vanilla, advrank:defense, experimental defense

# Classification
from . import cslp
from . import cc2f2
from . import clenet
from . import cres18

# Ranking / Metric Learning
# [group C2F2]
from . import rc2f2
from . import rc2f2d
from . import rc2f2df
from . import rc2f2e
from . import rc2f2p
from . import rc2f2px
from . import rc2f2pa
from . import rc2f2pf
from . import rc2f2t
from . import rc2f2a
from . import rc2f2r
from . import rc2f2m
from . import rc2f2amd
from . import rc2f2ramd
from . import rc2f2amdsemi

# [group LeNet]
from . import rlenet
from . import rlenetd
from . import rlenete

# [group MnasNet]
from . import rmnas05
from . import rmnas10
from . import rmnas10d
from . import rmnas10p

# [group ResNet]
from . import rres18
from . import rres18d
from . import rres18df
from . import rres18e
from . import rres18p
from . import rres18px
from . import rres18pf
from . import rres18t
from . import rres18r
from . import rres50
from . import rres50d
from . import rres50p

# [group EfficientNet]
from . import reffb0
from . import reffb0p

# [group Inception-BN (v2)]
from . import ribn
from . import ribnd
from . import ribnp

# Hybrid Metric+Classification models
from . import hc2f2
from . import hres18
