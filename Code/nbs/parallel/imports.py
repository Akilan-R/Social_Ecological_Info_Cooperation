import numpy as np
import matplotlib.pyplot as plt
import copy

from pyCRLD.Environments.SocialDilemma import SocialDilemma
from pyCRLD.Environments.EcologicalPublicGood import EcologicalPublicGood

from pyCRLD.Agents.StrategyActorCritic import stratAC
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Agents.POStrategyActorCritic_eps import POstratAC_eps


from pyCRLD.Utils import FlowPlot as fp
from fastcore.utils import *
from jax import jit
import jax.numpy as jnp
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
from functools import partial

from nbdev.showdoc import show_doc
from scipy.stats import kstest

from scipy.stats import qmc
import itertools as it
import pandas as pd
import plotly.graph_objects as go

import time
from datetime import timedelta
import math
from multiprocessing import Pool
import ast

global_seed = 41
np.random.seed(global_seed)