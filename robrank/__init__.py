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

from . import utils
from . import configs
from . import datasets
from . import models
from . import losses
from . import attacks
from . import defenses
from . import cmdline
import rich.traceback
rich.traceback.install()

r'''
Naming conventions for `robrank.models` and `robrank.configs`

```
<task><backbone><modifier>
```

* `task` in {c, r, h} for classification, ranking, hybrid
* `backbone` e.g. lenet, res18, mnas10
* `moddifier` in {, d, e, p} for vanilla, advrank:defense, ses, and pnp defenses

For example

* `cc2f2` stands for classification, c2f2
* `rc2f2` stands for ranking, c2f2
* `rc2f2d` stands for ranking, c2f2, with advrank:defense (EST)
'''
