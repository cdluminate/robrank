# Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>
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

# NOTE: you can write some targets like the following "legion" in
# Makefile.private for convenient code synchronization to your own
# servers without leaking secret.

-include ../../Makefile.private
-include ../Makefile.private
-include Makefile.private

RSYNC_DEST = ~/2021-robrank/

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

clean:
	-find . -name __pycache__ -exec rm -rfv '{}' +

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
	pytest -v -x  # -s # XXX: flag -s enables stdout

runtest:
	$(MAKE) -C robrank runtest

legion:
	#rsync -rvlz . 169.254.169.219:~/Git/2021-robrank/ $(RSYNC_FLAGS)
	rsync -rvlz . 192.168.1.106:$(RSYNC_DEST) $(RSYNC_FLAGS)

train-mnist:
	python3 bin/train.py -C mnist:rc2f2:ptripletN   # Vanilla
	python3 bin/train.py -C mnist:rc2f2d:ptripletN  # EST
	python3 bin/train.py -C mnist:rc2f2p:ptripletN  # ACT

train-fashion:
	python3 bin/train.py -C fashion:rc2f2:ptripletN   # Vanilla
	python3 bin/train.py -C fashion:rc2f2d:ptripletN  # EST
	python3 bin/train.py -C fashion:rc2f2p:ptripletN  # ACT

train-cub:
	python3 bin/train.py -C cub:rres18:ptripletN   # Vanilla
	python3 bin/train.py -C cub:rres18d:ptripletN  # EST
	python3 bin/train.py -C cub:rres18p:ptripletN  # ACT

train-cars:
	python3 bin/train.py -C cars:rres18:ptripletN   # Vanilla
	python3 bin/train.py -C cars:rres18d:ptripletN  # EST
	python3 bin/train.py -C cars:rres18p:ptripletN  # ACT

train-sop:
	python3 bin/train.py -C sop:rres18:ptripletN   # Vanilla
	python3 bin/train.py -C sop:rres18d:ptripletN  # EST
	python3 bin/train.py -C sop:rres18p:ptripletN  # ACT

fig2real-est:
	python3 bin/train.py -C fashion:rc2f2dco:ptripletN
	mv cosine_only_stat.json cosine_only_stat-est.json
	python3 tools/curve.py fig2real --file cosine_only_stat-est.json
	mv fig2real.svg fig2real-est.svg
	inkscape -o fig2real-est.pdf fig2real-est.svg
	pdfcrop fig2real-est.pdf fig2real-est.pdf

fig2real-act:
	python3 bin/train.py -C fashion:rc2f2pco:ptripletN
	mv cosine_only_stat.json cosine_only_stat-act.json
	python3 tools/curve.py fig2real --file cosine_only_stat-act.json
	mv fig2real.svg fig2real-act.svg
	inkscape -o fig2real-act.pdf fig2real-act.svg
	pdfcrop fig2real-act.pdf fig2real-act.pdf

fig3real-est:
	python3 bin/train.py -C fashion:rc2f2dbe:ptripletN
	mv batcheff_only_stat.json batcheff_only_stat-est.json
	python3 tools/curve.py fig3real --file batcheff_only_stat-est.json
	mv fig3real.svg fig3real-est.svg
	inkscape -o fig3real-est.pdf fig3real-est.svg
	pdfcrop fig3real-est.pdf fig3real-est.pdf

fig3real-act:
	python3 bin/train.py -C fashion:rc2f2pbe:ptripletN
	mv batcheff_only_stat.json batcheff_only_stat-act.json
	python3 tools/curve.py fig3real --file batcheff_only_stat-act.json
	mv fig3real.svg fig3real-act.svg
	inkscape -o fig3real-act.pdf fig3real-act.svg
	pdfcrop fig3real-act.pdf fig3real-act.pdf
