import json
import re
import argparse
import sys
import os
from termcolor import cprint, colored
import glob
import csv


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
                        last['NMI'] = f'{value:.2f}'
                    if 'r@1' in tag:
                        last['r@1'] = f'{100*value:.1f}'
                    if 'r@2' in tag:
                        last['r@2'] = f'{100*value:.1f}'
                    if 'mAP' in tag:
                        last['mAP'] = f'{value:.2f}'
                    for k in ('r@1', 'r@2', 'mAP', 'NMI'):
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
            print(k, v)


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
        for (k, v) in j.items():
            cprint(k, 'blue')
            if k.startswith('CA'):
                ca = tuple(x for x in v[-1].keys() if 'prank' in x)[0]
                print('DEBUG', ca)
                print(f'{100*float(v[-1][ca]):.1f}')
            elif k.startswith('SPQA'):
                qa = tuple(x for x in v[-1].keys() if ':prank' in x)[0]
                sp = tuple(x for x in v[-1].keys() if 'GTprank' in x)[0]
                print('DEBUG', qa, sp)
                print('DEBUG', 100 * float(v[-1][qa]), 100 * float(v[-1][sp]))
                print(
                    f'{100.*float(v[-1][qa]):.1f}',
                    f'{100.*float(v[-1][sp]):.1f}')
        #print(kfind(j.keys(), 'CA', 'pm=+', 'eps=0.3'))
        elo = 'eps=0.0' if any(
            x in ag.json for x in (
                'mnist', 'fashion')) else 'eps=0.00'
        ehi = 'eps=0.3' if any(
            x in ag.json for x in (
                'mnist', 'fashion')) else 'eps=0.06'

        # export CA+
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'CA', 'pm=+', 'W=1:', elo)[0]][1]['CA+:prank'])))
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'CA', 'pm=+', 'W=1:', ehi)[0]][1]['CA+:prank'])))
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'CA', 'pm=+', 'W=10:', elo)[0]][1]['CA+:prank'])))
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'CA', 'pm=+', 'W=10:', ehi)[0]][1]['CA+:prank'])))
        # export CA-
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'CA', 'pm=-', 'W=1:', elo)[0]][1]['CA-:prank']))
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'CA', 'pm=-', 'W=1:', ehi)[0]][1]['CA-:prank']))
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'CA', 'pm=-', 'W=10:', elo)[0]][1]['CA-:prank']))
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'CA', 'pm=-', 'W=10:', ehi)[0]][1]['CA-:prank']))
        # export QA+
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'QA', 'pm=+', 'M=1:', elo)[0]][1]['SPQA+:prank'])))
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'QA', 'pm=+', 'M=1:', ehi)[0]][1]['SPQA+:prank'])))
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'QA', 'pm=+', 'M=10:', elo)[0]][1]['SPQA+:prank'])))
        self.brief.append("%.1f" % (
            100 * min(0.5, j[kfind(j.keys(), 'QA', 'pm=+', 'M=10:', ehi)[0]][1]['SPQA+:prank'])))
        rgt = max([j[x][1]['SPQA+:GTprank']
                   for x in kfind(j.keys(), 'QA', 'pm=+')])
        self.brief.append("%.1f" % (100 * rgt))
        # export QA-
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'QA', 'pm=-', 'M=1:', elo)[0]][1]['SPQA-:prank']))
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'QA', 'pm=-', 'M=1:', ehi)[0]][1]['SPQA-:prank']))
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'QA', 'pm=-', 'M=10:', elo)[0]][1]['SPQA-:prank']))
        self.brief.append("%.1f" % (
            100 * j[kfind(j.keys(), 'QA', 'pm=-', 'M=10:', ehi)[0]][1]['SPQA-:prank']))
        rgt = max([j[x][1]['SPQA-:GTprank']
                   for x in kfind(j.keys(), 'QA', 'pm=-')])
        self.brief.append("%.1f" % (100 * rgt))


def autodiscoverjsontf(logdir: str):
    path = os.path.join(logdir, 'lightning_logs/version_*')
    ndir = nsort(glob.glob(path), r'.*version_(\d+)')[0]

    try:
        path = os.path.join(ndir, 'events.out.tfevents.*')
        ntfe = glob.glob(path)[0]
    except IndexError:
        print('cannot find any tfevent for the latest version')
        exit(1)
    cprint(f'* Automatically discovered latest tfevent .. {ntfe}', 'cyan')
    tfe = TFdump(['-f', ntfe])
    print('BEST', tfe.best)

    try:
        path = os.path.join(ndir, 'checkpoints/epoch=*.json')
        nchk = nsort(glob.glob(path), r'.*epoch=(\d+)')[0]
    except IndexError:
        print('cannot find any json for the latest version')
        exit(2)
    cprint(f'* Automatically discovered latest checkpoint .. {nchk}', 'cyan')
    J = Jdump(['-j', nchk])

    dataset, model, loss = re.match(
        r'.*?logs_(\w+)-(\w+)-(\w+)', logdir).groups()
    print('-- CSV/TeX --------------------------------')
    w = csv.writer(sys.stdout, delimiter='&')
    w.writerow([
        dataset,
        model,
        loss,
        "%.1f" % (tfe.best['r@1'] * 100),
        "%.1f" % (tfe.best['r@2'] * 100),
        "%.2f" % (tfe.best['mAP']),
        "%.2f" % (tfe.best['NMI']),
        *J.brief,
    ])
    #print(kfind(j.keys(), 'CA', 'pm=+', 'eps=0.3'))
    with open('pjswipe.csv', 'a+') as f:
        w = csv.writer(f, delimiter='&')
        w.writerow([
            dataset,
            model,
            loss,
            "%.1f" % (tfe.best['r@1'] * 100),
            "%.1f" % (tfe.best['r@2'] * 100),
            "%.2f" % (tfe.best['mAP']),
            "%.2f" % (tfe.best['NMI']),
            *J.brief,
        ])


if __name__ == '__main__':
    autodiscoverjsontf(sys.argv[1])
