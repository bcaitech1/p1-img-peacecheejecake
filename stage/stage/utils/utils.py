import os
from datetime import datetime
from glob import glob
import numpy as np

import scipy.stats

import torch
import torch.nn as nn

from .seed import *


##########################################
# UTILITIES ##############################
##########################################

def datetime_str(datetime: datetime, include_date=False, include_time=True, include_decimal=False):
    '''Convert datetime object to str
    '''
    date, time = str(datetime).split()
    time, decimal = time.split('.')
    datetime_str = ''
    if include_date: datetime_str += date
    if include_time: datetime_str += f' {time}'
    if include_decimal: datetime_str += f'.{decimal[:3]}'
    return datetime_str


def time_str(millisecond: float, include_decimal=False):
    r'''Convert time to str
    Args:
        millisecond (float): elapsed time recorded by torch.cuda.Event
        include_decimal (bool): whether include decimal points to second
    '''
    second, decimal = divmod(int(millisecond), 1000)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    decimal = str(decimal).rjust(3, '0')

    time_str = f'{minute:02d}:{second:02d}'
    if hour > 0:
        time_str = f'{hour}:' + time_str

    if include_decimal:
        time_str += f'.{decimal}'
    
    return time_str


def filename_from_datetime(datetime: datetime):
    '''Create filename from datetime
    '''
    filename = '_'.join(str(datetime).split(':'))
    filename = '-'.join(filename.split())
    
    return filename


def empty_logs():
    for log in glob('/opt/ml/output/logs/*.csv'):
        os.remove(log)


def empty_checkpoints():
    for model in glob('/opt/ml/output/models/*.pth'):
        os.remove(model)


def cut_pretrained_out_features(model, num_classes=18, bias=True, choose=False):
    out_layer = model.fc
    if out_layer.out_features > num_classes:
        if choose:
            indices = np.random.randint(0, out_layer.out_features, num_classes)
            out_layer.weight = nn.Parameter(out_layer.weight[indices])
            if bias: out_layer.bias = nn.Parameter(out_layer.bias[indices])
            out_layer.out_features = num_classes
        else:
            model.fc = nn.Linear(out_layer.in_features, num_classes)


try:
    load_state_dict_from_url = torch.hub.load_state_dict_from_url
except:
    load_state_dict_from_url = torch.utils.model_zoo.load_url

