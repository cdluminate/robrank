HEADER = '''\
Copyright (C) 2019-2021, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.\
'''

from typing import *
from enum import Enum
import os
import argparse
import sys
import re
import rich
c = rich.get_console()


class State(Enum):
    INIT = 0
    FIRST_QUOTE_START = 1
    FIRST_QUOTE = 2
    FIRST_QUOTE_END = 3
    TRAP = 4


def has_copyright_header(lines: List[str], filtering=False) -> bool:
    '''
    Does the given lines contain a copyright header?
    '''
    needle = re.compile(r'.*copyright.*', re.IGNORECASE)
    tquote = re.compile(r'.*[\'"]{3}.*')
    st, ret, flines = State.INIT, False, []
    for line in lines:
        # change status
        if tquote.match(line) and st == State.INIT:
            st = State.FIRST_QUOTE_START
        elif st == State.FIRST_QUOTE_START:
            st = State.FIRST_QUOTE
        elif tquote.match(line) and st == State.FIRST_QUOTE:
            st = State.FIRST_QUOTE_END
        elif st == State.FIRST_QUOTE_END:
            st = State.TRAP
        elif st in (State.INIT, State.FIRST_QUOTE, State.TRAP):
            pass
        else:
            raise Exception(line, st)
        # actions
        if st == State.FIRST_QUOTE and needle.match(line):
            if not filtering:
                return True
            else:
                ret = True
        if st in (State.INIT, State.TRAP):
            flines.append(line)
    if not filtering:
        return False
    else:
        ret = True if ret else False
    return ret, flines


def edit_python_file(fpath: str, ag: vars) -> None:
    '''
    Edit the given python file.
    '''
    with open(ag.file, 'rt') as f:
        lines = f.readlines()
        #lines = [line.strip() for line in lines]
    flag, flines = has_copyright_header(lines, filtering=True)
    if not flag and not ag.reverse:
        # does not have header, not reverse operation
        lines = ["'''\n"] + [x+'\n' for x in HEADER.split('\n')] + ["'''\n"] + lines
    elif flag and ag.reverse:
        # has header, reverse operation
        lines = flines
    # how to write the output
    if ag.inplace:
        with open(ag.file, 'wt') as f:
            f.writelines(lines)
    else:
        for l in lines:
            print(l, end='')
    if ag.verbose:
        c.print(fpath, ':', 'has_copyright_header=', flag)


if __name__ == '__main__':
    ag = argparse.ArgumentParser()
    ag.add_argument('-f', '--file', type=str, required=True,
            help='Specify the python file on which we operate.')
    ag.add_argument('-i', '--inplace', action='store_true',
            help='Do we edit the given python file inplace?')
    ag.add_argument('-R', '--reverse', action='store_true',
            help='Do we remove the header instead?')
    ag.add_argument('-v', '--verbose', action='store_true',
            help='Toggle verbose.')
    ag = ag.parse_args()

    # edit.
    edit_python_file(ag.file, ag)
