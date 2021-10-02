#!/bin/env bash
PY=robrank/models/autogen/autogen.py
cd $(dirname $PY)
python3 $(basename $PY)
