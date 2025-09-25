# marble/tasks/MTT/probe.py
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import mir_eval
import torchmetrics
from torchmetrics import Metric, MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.MTGMood.probe import ProbeAudioTask as _ProbeAudioTask


class ProbeAudioTask(_ProbeAudioTask):
    pass