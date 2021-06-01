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
	-pyflakes3 .
	fdfind $(FDFIND_FLAGS) -e py -x autopep8 -avi
	$(MAKE) -C robrank

doc:
	# clone this repo to the parent directory (..) before building doc
	# https://github.com/jothepro/doxygen-awesome-css.git
	doxygen

pylint:
	pylint -j8 *.py
	#pylint -j4 robrank
	pylint -j8 -d C,R robrank/

install_deps:
	pip install -r requirements.txt

download:
	python3 tools/download.py

pytest:
	pytest -v -x

runtest:
	$(MAKE) -C robrank runtest

legion:
	#rsync -rvlz . 169.254.169.219:~/Git/2021-robrank/ $(RSYNC_FLAGS)
	rsync -rvlz . 192.168.1.106:~/Git/2021-robrank/ $(RSYNC_FLAGS)

train-attack-defense-mnist-rc2f1-ptripletN:
	python3 train.py -C mnist:rc2f2:ptripletN
	python3 swipe.py -p pami28 -c logs_mnist-rc2f2-ptripletN
	python3 tools/pjswipe.py logs_mnist-rc2f2-ptripletN
	python3 train.py -C mnist:rc2f2d:ptripletN
	python3 swipe.py -p pami28 -c logs_mnist-rc2f2d-ptripletN
	python3 tools/pjswipe.py logs_mnist-rc2f2d-ptripletN

train-attack-defense-cub-rres18-ptripletN:
	python3 train.py -C cub:rres18:ptripletN
	python3 swipe.py -p pami224 -c logs_cub-rres18-ptripletN
	python3 advrank.py -D cuda -A SPQA:pm=-:M=1:eps=0.03137:alpha=0.011764:pgditer=32 -C $(ckpt_cub_rres18_ptripletN) --maxepoch 10
