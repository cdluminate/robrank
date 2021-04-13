#!/usr/bin/fish
python3 advrank.py -v -A CA:pm=+:W=1:eps=0.300000:alpha=0.011764:pgditer=32 -C $argv[1]
#python3 advrank.py -v -A CA:pm=+:W=1:eps=0.062745:alpha=0.011764:pgditer=32 -C $argv[1]
