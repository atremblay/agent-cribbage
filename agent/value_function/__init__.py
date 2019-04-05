import sys
from glob import glob
from os.path import basename, dirname, join, commonprefix
import os

pwd = dirname(__file__)

sys.path.append(pwd)
common = commonprefix([pwd, os.getcwd()])
parent_script = '.'.join(pwd[len(common)+1:].split(os.sep))
for x in glob(join(pwd, '*.py')):
    if not x.startswith('__'):
        __import__(parent_script + '.' + basename(x)[:-3], globals(), locals())

sys.path.remove(pwd)