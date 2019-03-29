import sys
from glob import glob
from os.path import basename, dirname, join, commonprefix
import os

pwd = dirname(__file__)

sys.path.append(pwd)
common = commonprefix([pwd, os.getcwd()])
parent = '.'.join(pwd[len(common)+1:].split(os.sep)[1:])
for x in glob(join(pwd, '*.py')):
    if not x.startswith('__'):
        __import__(parent+'.' + basename(x)[:-3], globals(), locals())

sys.path.remove(pwd)


