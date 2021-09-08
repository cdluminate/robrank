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
from . import rc2f2
from .. import utils


class Model(rc2f2.Model):
    is_advtrain_amd = True

    def post_init_hook(self):
        utils.warn('reducing advtrain_pgditer to 7 or it will collapse')
        utils.warn('this warning is irrelevant to attack')
        self.config.advtrain_pgditer = 7
