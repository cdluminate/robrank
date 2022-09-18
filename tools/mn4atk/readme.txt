To obtain the data:

1. first evaluate model with the qccurve28 profile.

python3 bin/swipe.py -p qccurve28 -C logs_fashion-rc2f2d-ptripletN/lightning_logs/version_0/checkpoints/epoch=7-step=1880.ckpt -v

2. export JTYPE=qccurve28 to specify that we want to dump *.qccurve28.json

3. dump json into txt files

mkdir -p tools/mn4atk/
python3 tools/pjswipe.py logs_fashion-rc2f2-ptripletN/ | sed -e 's/\x1b\[[0-9;]*m//g' > tools/mn4atk/fashion.txt 
python3 tools/pjswipe.py logs_fashion-rc2f2d-ptripletN/ | sed -e 's/\x1b\[[0-9;]*m//g' > tools/mn4atk/fashion-d.txt 
python3 tools/pjswipe.py logs_fashion-rc2f2p-ptripletN/ | sed -e 's/\x1b\[[0-9;]*m//g' > tools/mn4atk/fashion-p.txt 
