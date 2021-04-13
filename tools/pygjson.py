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
  "CA:pm=+:W=1": [
    {
      "loss": 0.18415331840515137,
      "CA+:prank": 0.4857479070975755
    },
    {
      "loss": 0.09100765486558278,
      "CA+:prank": 0.2500986921288073,
      "embshift": 0.05232645322879156
    }
  ],
  "CA:pm=-:W=1": [
    {
      "loss": 0.9074706633885702,
      "CA-:prank": 0.018794270426407456
    },
    {
      "loss": 0.7722941637039185,
      "CA-:prank": 0.08609713357289632,
      "embshift": 0.07193221648534139
    }
  ],
  "SPQA:pm=+:M=1": [
    {
      "loss": 0.17758294443289438,
      "SPQA+:prank": 0.47683255208333336,
      "SPQA+:GTprank": 0.0003
    },
    {
      "loss": 0.0830311228831609,
      "SPQA+:prank": 0.25804609375000004,
      "SPQA+:GTprank": 0.007887812644677857,
      "embshift": 0.04966878270109495
    }
  ],
  "SPQA:pm=-:M=1": [
    {
      "loss": 0.9048512578010559,
      "SPQA-:prank": 0.00518671875,
      "SPQA-:GTprank": 0.00030312500031044086
    },
    {
      "loss": 0.7533260981241862,
      "SPQA-:prank": 0.0351328125,
      "SPQA-:GTprank": 0.008336718889387945,
      "embshift": 0.0686750238140424
    }
  ]
}
'''


#formatter = pygments.formatters.terminal256.Terminal256Formatter()
formatter = pygments.formatters.terminal.TerminalFormatter()
#formatter = pygments.formatters.terminal256.TerminalTrueColorFormatter,

print(highlight(code, PythonLexer(), formatter))
#print(highlight(code, JavascriptLexer(), formatter))
