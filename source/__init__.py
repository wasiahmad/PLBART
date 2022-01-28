import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from . import translation
from . import multi_translation
from . import sentence_prediction
from . import completion
