'''
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

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
import os
import sys
import glob
import re
import argparse
import json
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

# entry names, 11 in total
#  r1 | cap, cam, qap, qam, tma | esd, esr, ltm, gtm, gtt

dots = []
for (_, jf) in ejsons:
    with open(jf, 'rt') as f:
        j = json.load(f)
    entry = [-1.0] * 11
    for (k, v) in j.items():
        if k.startswith('GTT:'):
            entry[10] = v[1]['retain@4'] * 100
        elif k.startswith('GTM:'):
            entry[0] = v[0]['r@1'] * 100
            entry[9] = v[1]['r@1'] * 100
        elif k.startswith('LTM:'):
            entry[8] = v[1]['r@1'] * 100
        elif k.startswith('ES:'):
            entry[7] = v[1]['r@1']
            entry[6] = v[1]['embshift']
        elif k.startswith('TMA:'):
            entry[5] = v[1]['Cosine-SIM']
        elif k.startswith('QA:pm=-'):
            entry[4] = v[1]['QA-:prank'] * 100
        elif k.startswith('QA:pm=+'):
            entry[3] = v[1]['QA+:prank'] * 100
        elif k.startswith('CA:pm=-'):
            entry[2] = v[1]['CA-:prank'] * 100
        elif k.startswith('CA:pm=+'):
            entry[1] = v[1]['CA+:prank'] * 100
        else:
            print('warning: unknown key', k)
    dots.append(entry)
c.print(dots)
