import os
import sys
import glob
import re
import argparse
import rich
c = rich.get_console()

ag = argparse.ArgumentParser(description='''
make plots after finishing tools/swipeall.bash
''')
ag.add_argument('-d', '--directory', type=str, required=True,
    help='directory containing json files')
# exmple: 'logs_cub-rres18p-ptripletN/lightning_logs/version_1/'
ag = ag.parse_args()
print(ag)

if not os.path.exists(ag.directory):
    raise FileNotFoundError(ag.directory)

jsons = glob.glob(os.path.join(ag.directory, '**/*.json'), recursive=True)
# 'logs_cub-rres18p-ptripletN/lightning_logs/version_1/checkpoints/epoch=4-step=135.ckpt.rob224.json',
ejsons = [ (re.match(r'.*epoch=(\d+)-step.*', j).groups()[0],
           j) for j in jsons]
ejsons = sorted(ejsons, key=lambda x: int(x[0]))
c.print('debug: ejsons:', ejsons)


