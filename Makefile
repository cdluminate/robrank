# Copyright (C) 2019-2021, Mo Zhou <cdluminate@gmail.com>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

RSYNC_FLAGS = \
	--exclude=.git \
	--exclude=__pycache__  \
	--exclude=logs_* \
	--exclude=exp.lyx*

FDFIND_FLAGS = \
	$(RSYNC_FLAGS)

main:
	fdfind $(FDFIND_FLAGS) -e py -x autopep8 -ai
	$(MAKE) -C robrank

install_deps:
	pip install -r requirements.txt

download:
	python3 tools/download.py

pytest:
	pytest -v -x

runtest:
	$(MAKE) -C robrank runtest
