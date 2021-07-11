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
