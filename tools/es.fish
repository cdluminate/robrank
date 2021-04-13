#!/usr/bin/fish
python3 advrank.py -v -A ES:eps=0.3:alpha=(math 2/255):pgditer=32 -C $argv[1]
