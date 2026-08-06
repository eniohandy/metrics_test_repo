import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from imports import *

## Faithfulness está em ragas.metrics.collections e não em ragas.metrics
### troque o nome da métrica e teste a importação

from ragas.metrics.collections import AnswerAccuracy 


print ("importado")