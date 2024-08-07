import math
import torch
import torch.nn as nn
import torch.nn.functional as Fd
import numpy as np
import pandas as p
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
import collections
from torch.autograd import Variable
from torch.nn import DataParallel
from math import sqrt
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from typing import Optional, Tuple, Union
from collections import OrderedDict
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns