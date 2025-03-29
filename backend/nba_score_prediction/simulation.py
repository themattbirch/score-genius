# backend/nba_score_prediction/simulation.py

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import time
import math
from functools import wraps
import scipy.stats as stats
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Union, Any