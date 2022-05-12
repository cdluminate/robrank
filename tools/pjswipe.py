#!/bin/python3
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
from collections import namedtuple
from termcolor import cprint, colored
import argparse
import csv
import glob
import json
import numpy as np
import os
import re
import sys
import rich
c = rich.get_console()


Scores = namedtuple(
    'Scores', [
        'cap', 'cam', 'qap', 'qam', 'tma',
        'esd', 'esr', 'ltm', 'gtm', 'gtt'])


def _c(line: str):
    return tuple(map(float, line.split('\t')))


def ersnormalize(ss: Scores):
    a = np.zeros(10)
    a[0] = ss.cap * 2.0
    a[1] = 100.0 - ss.cam
    a[2] = ss.qap * 2.0
    a[3] = 100.0 - ss.qam
    a[4] = 100.0 * (1.0 - ss.tma)
    a[5] = 100.0 * (2.0 - ss.esd) / 2
    a[6] = ss.esr
    a[7] = ss.ltm
    a[8] = ss.gtm
    a[9] = ss.gtt
    return a


def ers(ss: Scores):
    '''
    Calculate ERS from a given Scores instance.
    '''
    c.print('Raw Scores', ss)
    a = ersnormalize(ss)
    c.print('Rescaled Scores', a)
    return f'{a.mean():.1f}'


def nsort(L: list, R: str):
    '''
    sort list L by the key:int matched from regex R, descending.
    '''
    assert(all(re.match(R, item) for item in L))
    nL = [(int(re.match(R, item).groups()[0]), item) for item in L]
    nL = sorted(nL, key=lambda x: x[0], reverse=True)
    return [x[-1] for x in nL]


class TFdump:
    '''
    Dump a tensorboard tfevent binary file
    '''
    best = {}

    def __init__(self, *args):
        ag = argparse.ArgumentParser()
        ag.add_argument('-f', '--file', type=str, required=True)
        ag = ag.parse_args(*args)

        from tensorflow.python.summary.summary_iterator import summary_iterator
        last = dict()
        for s in summary_iterator(ag.file):
            # print(s)
            if len(s.summary.value) > 0:
                tag = s.summary.value[0].tag
                if 'Valid' in tag:
                    cprint(s.summary.value[0].tag, 'red', end=' ')
                    cprint(f'{s.step}', 'yellow', end=' ')
                    cprint(s.summary.value[0].simple_value, 'blue')
                    value = s.summary.value[0].simple_value
                    if 'NMI' in tag:
                        last['NMI'] = f'{100*value:.1f}'
                    elif 'r@1' in tag:
                        last['r@1'] = f'{100*value:.1f}'
                    elif 'r@2' in tag:
                        last['r@2'] = f'{100*value:.1f}'
                    elif 'mAP@R' in tag:
                        last['mAP@R'] = f'{100*value:.1f}'
                    elif ('mAP' in tag) and not ('@R' in tag):
                        last['mAP'] = f'{100*value:.1f}'
                    else:
                        pass
                    for k in ('r@1', 'r@2', 'mAP@R', 'mAP', 'NMI'):
                        if k not in tag:
                            continue
                        if k not in self.best:
                            self.best[k] = float(value)
                        else:
                            # NOTE: we should use the last value instead of the
                            # best value, because the adversarial attack is evaluated
                            # on the last checkpoint.
                            self.best[k] = float(value)
                            if float(value) > self.best[k]:
                                self.best[k] = float(value)
        for (k, v) in last.items():
            print('LAST', k, v, f'(BEST is {100*self.best[k]:.1f}')


def kfind(L: list, *args):
    result = [x for x in L if all(y in x for y in args)]
    return result


class Jdump:
    '''
    dump a json file
    '''
    json = None
    brief = []

    def __init__(self, *args):
        ag = argparse.ArgumentParser()
        ag.add_argument('-j', '--json', type=str, required=True)
        ag = ag.parse_args(*args)

        with open(ag.json, 'rt') as f:
            j = json.load(f)
        self.json = j
        if 'rob' in ag.json:
            Score = Scores(*((0.0,) * 10))
        for (k, v) in j.items():
            cprint(k, 'blue')
            if k.startswith('CA'):
                ca = tuple(x for x in v[-1].keys() if 'prank' in x)[0]
                #print('DEBUG', ca)
                print(f'PCTL {100*float(v[-1][ca]):.1f} / 100.0')
                if 'rob' in ag.json:
                    if 'pm=+' in k:
                        Score = Score._replace(cap=100 * float(v[-1][ca]))
                    elif 'pm=-' in k:
                        Score = Score._replace(cam=100 * float(v[-1][ca]))
            elif k.startswith('QA'):
                qa = tuple(x for x in v[-1].keys() if 'prank' in x)[0]
                print(f'PCTL {100*float(v[-1][qa]):.1f} / 100.0')
                if 'rob' in ag.json:
                    if 'pm=+' in k:
                        Score = Score._replace(qap=100 * float(v[-1][qa]))
                    elif 'pm=-' in k:
                        Score = Score._replace(qam=100 * float(v[-1][qa]))
            elif k.startswith('SPQA'):
                qa = tuple(x for x in v[-1].keys() if ':prank' in x)[0]
                sp = tuple(x for x in v[-1].keys() if 'GTprank' in x)[0]
                #print('DEBUG', qa, sp)
                #print('DEBUG', 100 * float(v[-1][qa]), 100 * float(v[-1][sp]))
                print(
                    f'PCTL {100.*float(v[-1][qa]):.1f} / 100.0',
                    f'PCTL {100.*float(v[-1][sp]):.1f} / 100.0')
            elif k.startswith('ES'):
                r1 = float(v[-1]['r@1'])
                es = float(v[-1]['embshift'])
                #print('DEBUG', f'{r1:.1f}/100', f'{es:.3f}/2')
                print(f'ES {es:.3f} / 2.000')
                print(f'R1 {r1:.1f} / 100.0')
                if 'rob' in ag.json:
                    Score = Score._replace(esd=es, esr=r1)
            elif k.startswith('TMA'):
                cs = float(v[-1]['Cosine-SIM'])
                print(f'COS {cs:.3f} / 1.000')
                if 'rob' in ag.json:
                    Score = Score._replace(tma=cs)
            elif k.startswith('LTM'):
                r1 = float(v[-1]['r@1'])
                print(f'R1 {100.*r1:.1f} / 100.0')
                if 'rob' in ag.json:
                    Score = Score._replace(ltm=100 * r1)
            elif k.startswith('GTM'):
                r1 = 100 * float(v[-1]['r@1'])
                #print('DEBUG', f'{r1:.1f}/100')
                print(f'R1 {r1:.1f} / 100.0')
                if 'rob' in ag.json:
                    Score = Score._replace(gtm=r1)
            elif k.startswith('GTT'):
                rt4 = 100 * float(v[-1]['retain@4'])
                print(f'RETAIN4 {rt4:.1f} / 100.0')
                if 'rob' in ag.json:
                    Score = Score._replace(gtt=rt4)
            else:
                raise NotImplementedError(k)
        #print(kfind(j.keys(), 'CA', 'pm=+', 'eps=0.3'))

        # get the ERS for robxxx.json
        if 'rob' in ag.json:
            # c.print(Score)
            c.print('Eventual ERS', ers(Score))


def autodiscoverjsontf(logdir: str):
    path = os.path.join(logdir, 'lightning_logs/version_*')
    ITH = int(os.getenv('ITH', '0'))
    ndir = nsort(glob.glob(path), r'.*version_(\d+)')[ITH]

    try:
        path = os.path.join(ndir, 'events.out.tfevents.*')
        ntfe = glob.glob(path)[0]
    except IndexError:
        print('cannot find any tfevent for the latest version')
        exit(1)
    cprint(f'* Automatically discovered latest tfevent .. {ntfe}', 'cyan')
    tfe = TFdump(['-f', ntfe])

    JTYPE = str(os.getenv('JTYPE', ''))
    EPH = int(os.getenv('EPH', 0))
    try:
        path = os.path.join(ndir, f'checkpoints/epoch=*{JTYPE}.json')
        nchk = nsort(glob.glob(path), r'.*epoch=(\d+)')[EPH]
    except IndexError:
        print('cannot find any json for the latest version')
        exit(2)
    cprint(f'* Automatically discovered latest checkpoint .. {nchk}', 'cyan')
    J = Jdump(['-j', nchk])


if __name__ == '__main__':
    print('''\
Usage: pjswipe.py xxx/logs_mnist-rc2f2-ptripletN

export JTYPE={rob28,rob224,pami28,pami224} to select json.
export ITH=int to select version
''')
    autodiscoverjsontf(sys.argv[1])
