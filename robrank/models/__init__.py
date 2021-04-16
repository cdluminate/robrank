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
from . import cc2f2
from . import clenet
from . import csres18

# Ranking / Metric Learning
from . import rc2f2
from . import rc2f2d
from . import rc2f2e
from . import rlenet
from . import rlenetd
from . import rlenete
from . import rmnas05
from . import rmnas10
from . import rmnas10d
from . import rres18
from . import rres18d
from . import rres18e
from . import rres50
from . import rres50d

# Hybrid Metric+Classification models
from . import hc2f2
from . import hres18
