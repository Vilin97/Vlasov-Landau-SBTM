import os
ROOT = os.path.expanduser('~/Vlasov-Landau-SBTM/')
DATA = os.path.join(ROOT, 'data')
PLOTS = os.path.join(DATA, 'plots')
MODELS = os.path.join(DATA, 'score_models')

# set higher image quality
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# set path
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))