#!/usr/bin/fish
python3 advrank.py -v -A SPQA:pm=-:M=1:eps=0.300000:alpha=0.011764:pgditer=32 -C $argv[1] -b16
#python3 advrank.py -v -A SPQA:pm=-:M=10:eps=0.300000:alpha=0.011764:pgditer=32 -C $argv[1] -b16
