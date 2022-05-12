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
import pygments.formatters.terminal
import pygments.formatters.terminal256
from pygments.lexers import JavascriptLexer
from pygments.lexers import PythonLexer
from pygments import highlight
code = '''
{
  "ES:eps=0.3:alpha=(math 2/255):pgditer=32": [
    {
      "loss": 0.0,
      "r@1": 99.47916666666667,
      "r@10": 99.73958333333333,
      "r@100": 100.0
    },
    {
      "loss": -30.71092478434245,
      "r@1": 75.0,
      "r@10": 88.80208333333333,
      "r@100": 94.01041666666667,
      "embshift": 0.22599885861078897
    }
  ],
}
'''


#formatter = pygments.formatters.terminal256.Terminal256Formatter()
formatter = pygments.formatters.terminal.TerminalFormatter()
#formatter = pygments.formatters.terminal256.TerminalTrueColorFormatter,

print(highlight(code, PythonLexer(), formatter))
#print(highlight(code, JavascriptLexer(), formatter))
