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
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools as ft
import json
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = 'Times New Roman'
rcParams['font.serif'] = 'Linux Libertine O'
rcParams['font.size'] = 14
rcParams['mathtext.fontset'] = 'cm'


def readr1(path: str) -> np.array:
    with open(path, 'rt') as f:
        data = f.readlines()
    data = list(filter(lambda x: 'r@1' in x, data))
    data = list(map(lambda x: float(x.strip().split()[-1]), data))
    data = np.array(data) * 100
    return data


class Curve:
    def svg(self, path: str):
        plt.savefig(path)


class ExpR1Curve(Curve):
    tsize = 24
    lsize = 20

    def __init__(self):

        plt.figure(figsize=[5 * 5, 4.8], tight_layout=True)

        ax = plt.subplot(1, 5, 1)
        plt.grid(True, linestyle=':')
        x = np.arange(0, 8) + 1
        plt.plot(x, readr1('expr1/mnist.txt'), marker='.', color='tab:gray')
        plt.plot(x, readr1('expr1/mnist-d.txt'), marker='v', color='tab:blue')
        plt.plot(x, readr1('expr1/mnist-p.txt'), marker='*', color='tab:red')
        plt.legend(['Vanilla', 'EST', 'ACT'], prop={'size': 17})
        plt.xlabel('Number of Epochs', size=self.lsize)
        plt.ylabel('Recall@1', size=self.lsize)
        plt.title('R@1 Curve on MNIST', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 5, 2)
        plt.grid(True, linestyle=':')
        x = np.arange(0, 8) + 1
        plt.plot(x, readr1('expr1/fashion.txt'), marker='.', color='tab:gray')
        plt.plot(
            x,
            readr1('expr1/fashion-d.txt'),
            marker='v',
            color='tab:blue')
        plt.plot(x, readr1('expr1/fashion-p.txt'), marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'])
        plt.xlabel('Number of Epochs', size=self.lsize)
        #plt.ylabel('Recall@1', size=self.lsize)
        plt.title('R@1 Curve on Fashion', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 5, 3)
        plt.grid(True, linestyle=':')
        x = (np.arange(0, 15) + 1) * 10
        plt.plot(x, readr1('expr1/cub.txt'), marker='.', color='tab:gray')
        plt.plot(x, readr1('expr1/cub-d.txt'), marker='v', color='tab:blue')
        plt.plot(x, readr1('expr1/cub-p.txt'), marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'],
        #           loc='center right', bbox_to_anchor=(1.0, 0.7))
        plt.xlabel('Number of Epochs', size=self.lsize)
        #plt.ylabel('Recall@1', size=self.lsize)
        plt.title('R@1 Curve on CUB', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 5, 4)
        plt.grid(True, linestyle=':')
        x = (np.arange(0, 15) + 1) * 10
        plt.plot(x, readr1('expr1/cars.txt'), marker='.', color='tab:gray')
        plt.plot(x, readr1('expr1/cars-d.txt'), marker='v', color='tab:blue')
        plt.plot(x, readr1('expr1/cars-p.txt'), marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'],
        #           loc='center right', bbox_to_anchor=(1.0, 0.7))
        plt.xlabel('Number of Epochs', size=self.lsize)
        #plt.ylabel('Recall@1', size=self.lsize)
        plt.title('R@1 Curve on CARS', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 5, 5)
        plt.grid(True, linestyle=':')
        x = (np.arange(0, 15) + 1) * 10
        plt.plot(x, readr1('expr1/sop.txt'), marker='.', color='tab:gray')
        plt.plot(x, readr1('expr1/sop-d.txt'), marker='v', color='tab:blue')
        plt.plot(x, readr1('expr1/sop-p.txt'), marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'],
        #           loc='center right', bbox_to_anchor=(1.0, 0.7))
        plt.xlabel('Number of Epochs', size=self.lsize)
        #plt.ylabel('Recall@1', size=self.lsize)
        plt.title('R@1 Curve on SOP', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)


class FashionR1Curve(Curve):
    def __init__(self):

        plt.figure(figsize=[6.4, 5.8], tight_layout=True)

        plt.grid(True, linestyle=':')
        x = np.arange(0, 8) + 1
        #plt.plot(x, readr1('far1/fashion.txt'), marker='.', color='tab:gray')
        plt.plot(x, readr1('far1/fashion-d.txt'), marker='v', color='tab:blue')
        plt.plot(x, readr1('far1/fashion-db.txt'),
                 marker='v', color='xkcd:dark blue')
        plt.plot(x, readr1('far1/fashion-p.txt'), marker='*', color='tab:red')
        plt.plot(x, readr1('far1/fashion-pb.txt'),
                 marker='*', color='xkcd:dark red')
        plt.plot(
            x,
            readr1('far1/fashion-r.txt'),
            marker='d',
            color='xkcd:cyan')
        plt.plot(x, readr1('far1/fashion-rb.txt'),
                 marker='d', color='xkcd:dark cyan')
        plt.plot(
            x,
            readr1('far1/fashion-o.txt'),
            marker='o',
            color='xkcd:bright pink')
        plt.legend([  # 'Vanilla',
            'EST', 'EST($\\beta$)',
            'ACT', 'ACT($\\beta$)',
            'REST', 'REST($\\beta$)',
            'SES'], prop={'size': 10})
        plt.xlabel('Number of Epochs', size=12)
        plt.ylabel('Recall@1', size=12)
        plt.title('R@1 Curve of Defense Methods on Fashion-MNIST', size=14)


class RobustnessCurve(Curve):
    lsize = 16
    bbox = (0.92, 0.18)

    def __init__(self):
        from pjswipe import Scores, ersnormalize

        data = np.loadtxt('rob/rob.txt')
        data = [ersnormalize(Scores(*x)) for x in data]
        data = [np.array((*x, x[0])) for x in data]
        print(data)

        categories = ['CA+', 'CA-', 'QA+', 'QA-', 'TMA', 'ES:D', 'ES:R',
                      'LTM', 'GTM', 'GTT', '']
        label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(data[0]))

        plt.figure(figsize=(5 * 5, 5 + 0.5), tight_layout=True)

        ax = plt.subplot(1, 5, 1, polar=True)
        ax.set_theta_zero_location('E')
        plt.grid(True, linestyle='-')
        plt.plot(label_loc, data[0], label='Vanilla', color='xkcd:black')
        ax.fill(label_loc, data[0], color='xkcd:black', alpha=0.25)
        plt.plot(label_loc, data[1], label='EST', color='tab:blue')
        ax.fill(label_loc, data[1], color='tab:blue', alpha=0.15)
        plt.plot(label_loc, data[2], label='ACT', marker='*', color='tab:red')
        ax.fill(label_loc, data[2], color='tab:red', alpha=0.10)
        plt.ylim((0, 100))
        plt.title('Normalized Scores on MNIST', size=22)
        lines, labels = plt.thetagrids(
            np.degrees(label_loc), labels=categories, size=16)
        plt.legend(
            loc='lower right',
            bbox_to_anchor=self.bbox,
            prop={
                'size': self.lsize})

        ax = plt.subplot(1, 5, 2, polar=True)
        ax.set_theta_zero_location('E')
        plt.grid(True, linestyle='-')
        plt.plot(label_loc, data[3], label='Vanilla', color='xkcd:black')
        ax.fill(label_loc, data[3], color='xkcd:black', alpha=0.25)
        plt.plot(label_loc, data[4], label='EST', color='tab:blue')
        ax.fill(label_loc, data[4], color='tab:blue', alpha=0.15)
        plt.plot(label_loc, data[5], label='ACT', marker='*', color='tab:red')
        ax.fill(label_loc, data[5], color='tab:red', alpha=0.10)
        plt.ylim((0, 100))
        plt.title('Normalized Scores on Fashion', size=22)
        lines, labels = plt.thetagrids(
            np.degrees(label_loc), labels=categories, size=16)
        #plt.legend(
        #    loc='lower right',
        #    bbox_to_anchor=self.bbox,
        #    prop={
        #        'size': self.lsize})

        ax = plt.subplot(1, 5, 3, polar=True)
        ax.set_theta_zero_location('E')
        plt.grid(True, linestyle='-')
        plt.plot(label_loc, data[6], label='Vanilla', color='xkcd:black')
        ax.fill(label_loc, data[6], color='xkcd:black', alpha=0.25)
        plt.plot(label_loc, data[7], label='EST', color='tab:blue')
        ax.fill(label_loc, data[7], color='tab:blue', alpha=0.15)
        plt.plot(label_loc, data[8], label='ACT', marker='*', color='tab:red')
        ax.fill(label_loc, data[8], color='tab:red', alpha=0.10)
        plt.ylim((0, 100))
        plt.title('Normalized Scores on CUB', size=22)
        lines, labels = plt.thetagrids(
            np.degrees(label_loc), labels=categories, size=16)
        #plt.legend(
        #    loc='lower right',
        #    bbox_to_anchor=self.bbox,
        #    prop={
        #        'size': self.lsize})

        ax = plt.subplot(1, 5, 4, polar=True)
        ax.set_theta_zero_location('E')
        plt.grid(True, linestyle='-')
        plt.plot(label_loc, data[9], label='Vanilla', color='xkcd:black')
        ax.fill(label_loc, data[9], color='xkcd:black', alpha=0.25)
        plt.plot(label_loc, data[10], label='EST', color='tab:blue')
        ax.fill(label_loc, data[10], color='tab:blue', alpha=0.15)
        plt.plot(label_loc, data[11], label='ACT', marker='*', color='tab:red')
        ax.fill(label_loc, data[11], color='tab:red', alpha=0.10)
        plt.ylim((0, 100))
        plt.title('Normalized Scores on CARS', size=22)
        lines, labels = plt.thetagrids(
            np.degrees(label_loc), labels=categories, size=16)
        #plt.legend(
        #    loc='lower right',
        #    bbox_to_anchor=self.bbox,
        #    prop={
        #        'size': self.lsize})

        ax = plt.subplot(1, 5, 5, polar=True)
        ax.set_theta_zero_location('E')
        plt.grid(True, linestyle='-')
        plt.plot(label_loc, data[12], label='Vanilla', color='xkcd:black')
        ax.fill(label_loc, data[12], color='xkcd:black', alpha=0.25)
        plt.plot(label_loc, data[13], label='EST', color='tab:blue')
        ax.fill(label_loc, data[13], color='tab:blue', alpha=0.15)
        plt.plot(label_loc, data[14], label='ACT', marker='*', color='tab:red')
        ax.fill(label_loc, data[14], color='tab:red', alpha=0.10)
        plt.ylim((0, 100))
        plt.title('Normalized Scores on SOP', size=22)
        lines, labels = plt.thetagrids(
            np.degrees(label_loc), labels=categories, size=16)
        #plt.legend(
        #    loc='lower right',
        #    bbox_to_anchor=self.bbox,
        #    prop={
        #        'size': self.lsize})


class Mnist4AttackCurve(Curve):
    lsize = 20
    tsize = 24

    def slurp(self, path: str) -> tuple:
        with open(path, 'rt') as f:
            data = [
                float(
                    x[1].strip().split()[1]) for x in enumerate(
                    f.readlines()) if x[0] %
                2 == 1]
        l = len(data) // 4
        cap = np.array(data[:l]).clip(None, 49.9)
        cam = np.array(data[l:2 * l])
        qap = np.array(data[2 * l:3 * l]).clip(None, 49.9)
        qam = np.array(data[3 * l:4 * l])
        return (cap, cam, qap, qam)

    def __init__(self):
        dv = self.slurp('mn4atk/fashion.txt')
        dd = self.slurp('mn4atk/fashion-d.txt')
        dp = self.slurp('mn4atk/fashion-p.txt')

        plt.figure(figsize=(4 * 5, 5), tight_layout=True)

        ax = plt.subplot(1, 4, 1)
        plt.grid(True, linestyle=':')
        x = ((np.arange(0, 11 + 1)) * 7) / 255.
        plt.plot(x, dv[0], marker='.', color='tab:gray')
        plt.plot(x, dd[0], marker='v', color='tab:blue')
        plt.plot(x, dp[0], marker='*', color='tab:red')
        plt.legend(['Vanilla', 'EST', 'ACT'], loc='lower left', prop={'size': 17})
        plt.xlabel('$\\varepsilon$', size=self.lsize)
        plt.ylabel('Rank Percentile', size=self.lsize)
        plt.title('CA+ on Fashion', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 4, 2)
        plt.grid(True, linestyle=':')
        x = ((np.arange(0, 11 + 1)) * 7) / 255.
        plt.plot(x, dv[1], marker='.', color='tab:gray')
        plt.plot(x, dd[1], marker='v', color='tab:blue')
        plt.plot(x, dp[1], marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'], loc='upper left')
        plt.xlabel('$\\varepsilon$', size=self.lsize)
        #plt.ylabel('Rank Percentile', size=self.lsize)
        plt.title('CA- on Fashion', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 4, 3)
        plt.grid(True, linestyle=':')
        x = ((np.arange(0, 11 + 1)) * 7) / 255.
        plt.plot(x, dv[2], marker='.', color='tab:gray')
        plt.plot(x, dd[2], marker='v', color='tab:blue')
        plt.plot(x, dp[2], marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'], loc='lower left')
        plt.xlabel('$\\varepsilon$', size=self.lsize)
        #plt.ylabel('Rank Percentile', size=self.lsize)
        plt.title('SP-QA+ on Fashion', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        ax = plt.subplot(1, 4, 4)
        plt.grid(True, linestyle=':')
        x = ((np.arange(0, 11 + 1)) * 7) / 255.
        plt.plot(x, dv[3], marker='.', color='tab:gray')
        plt.plot(x, dd[3], marker='v', color='tab:blue')
        plt.plot(x, dp[3], marker='*', color='tab:red')
        #plt.legend(['Vanilla', 'EST', 'ACT'], loc='lower right')
        plt.xlabel('$\\varepsilon$', size=self.lsize)
        #plt.ylabel('Rank Percentile', size=self.lsize)
        plt.title('SP-QA- on Fashion', size=self.tsize)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)


class Fig2RealCosHist(Curve):
    def __init__(self, jsonpath: str):
        with open(jsonpath, 'rt') as f:
            j = json.load(f)
        print(f'>_< loaded json array with {len(j)} numbers.')
        self.j = np.array(j)
        print('min', self.j.min(), 'mean', self.j.mean(), 'max', self.j.max())

        plt.figure(figsize=(6.4, 2.4), tight_layout=True)
        n, bins, patches = plt.hist(self.j, range=(-1.0, 1.0), bins=40, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        col = np.sqrt(1 - col)  # slightly change colormap
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', plt.cm.get_cmap('Blues')(c))
        plt.ylabel('Density')
        plt.xlabel('Cosine Similarity')
        plt.grid(True, linestyle=':')
        #plt.title('Gradient Consistency after EST Attack', y=-1.01)
        #plt.show()


class Fig3RealBatchEff(Curve):
    def __init__(self, jsonpath: str):
        with open(jsonpath, 'rt') as f:
            j = json.load(f)
        print(f'>_< loaded json array with {len(j)} numbers.')
        self.j = np.array(j)
        print('min', self.j.min(), 'mean', self.j.mean(), 'max', self.j.max())

        plt.figure(figsize=(6.4, 2.4), tight_layout=True)
        n, bins, patches = plt.hist(self.j, range=(-0.21,0.21), bins=40, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        #col = bin_centers - min(bin_centers)
        #col /= max(col)
        col = bin_centers
        col = col / 0.21
        #col = np.sqrt(1 - col)  # slightly change colormap
        col = np.sqrt(np.abs(col))
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', plt.cm.get_cmap('Reds')(c))
        plt.ylabel('Density')
        plt.xlabel('Change in $\\beta+d(q,c_p)-d(q,c_n)$')
        plt.grid(True, linestyle=':')
        #plt.title('Gradient Consistency after EST Attack', y=-1.01)
        #plt.show()



if __name__ == '__main__':

    ag = argparse.ArgumentParser()
    ag.add_argument('spec', help='specification')
    ag.add_argument('--file', '-f', type=str, default='', help='optional data file')
    ag = ag.parse_args()

    if ag.spec == 'expr1':
        p = ExpR1Curve()
        p.svg('expr1.svg')
    elif ag.spec == 'far1':
        p = FashionR1Curve()
        p.svg('far1.svg')
    elif ag.spec == 'rob':
        p = RobustnessCurve()
        p.svg('rob.svg')
    elif ag.spec == 'mn4atk':
        p = Mnist4AttackCurve()
        p.svg('mn4atk.svg')
    elif ag.spec == 'fig2real':
        assert ag.file, 'please specify json path using --file'
        p = Fig2RealCosHist(ag.file)
        p.svg('fig2real.svg')
    elif ag.spec == 'fig3real':
        assert ag.file, 'please specify json path using --file'
        p = Fig3RealBatchEff(ag.file)
        p.svg('fig3real.svg')
