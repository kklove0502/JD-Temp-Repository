#coding=utf-8

import traceback
from datetime import date
import numpy as np
import pandas as pd
import json
import math

from ..config import *


class MFUtilityError(Exception):
    def __init__(self, log_info):
        self.log_info = log_info

    def __str__(self):
        return str(self.log_info)


class MFInputError(MFUtilityError):
    pass


class InputLengthError(MFInputError):
    pass


class InputTypeError(MFInputError):
    pass


class InputConfigError(MFInputError):
    pass


class MFOutputError(MFUtilityError):
    pass


class MFNetworkError(MFUtilityError):
    pass


def raw_data_check(raw_data):
    if not isinstance(raw_data, (dict, pd.Series)):
        raise InputTypeError('Input raw_data is not dictionary or Series')
    if len(raw_data) <= 1:
        # print 'raw data length is ', len(raw_data)
        raise InputLengthError('Input raw_data length too short')
    
    if isinstance(raw_data, dict):
        items = raw_data.items
    elif isinstance(raw_data, pd.Series):
        items = raw_data.iteritems

    for key, value in items():
        if not isinstance(value, (int, float, np.int64)):
            raise InputTypeError('Input raw_data value is not value')
        if isinstance(key, unicode):
            key = str(key)
        if not isinstance(key, (str, unicode)):
            raise InputTypeError('Input raw_data key is not str')
        key = str(key)
        key_check(key)


def key_check(input_str):
    if len(input_str) != 6 and len(input_str) != 11:
        raise InputTypeError('Input raw_data key length error')
    if len(input_str) == 11:
        key = input_str[:6]
        end = input_str[6:]
        if end not in ['.XSHE', '.XSHG']:
            raise InputTypeError('Input raw_data key type error')
    else:
        key = input_str[:6]

    if not key.isdigit():
        raise InputTypeError('Input raw_data key type error')


def industry_check(industry_type):
    if industry_type not in ['long', 'short', 'day', 'SW1', 'SW2']:
        raise InputConfigError('Industry Type Config Error')


def risk_model_check(risk_model):
    if risk_model not in ['long', 'short', 'day']:
        raise InputConfigError('Risk Model Config Error')


def universe_check(universe):
    '''
    :param universe: universe type
    :return: mapped universe type
    '''
    map_dict = {'HS300': 'HSSLL', 'SH50': 'SZWL', 'SH180': 'SZYBL', 'ZZ500': 'ZZWLL'}
    if universe in map_dict:
        universe = map_dict[universe]
    if universe not in ['HSSLL', 'SZWL', 'SZYBL', 'SZZS', 'ZZWLL']:
        raise InputConfigError('Universe Config Error')
    return universe


def date_check(target_date):
    if not isinstance(target_date, str) and not isinstance(target_date, unicode):
        raise InputTypeError('Input Target Date Type Error')
    if len(target_date) not in [8, 10]:
        raise InputTypeError('Input Target Data Length Error')
    if date.today().strftime('%Y%m%d') <= target_date.replace('-', ''):
        raise InputTypeError('Input Target Date Config Error')


def risk_factor_check(exclude_factor_list):
    if not isinstance(exclude_factor_list, list):
        raise InputTypeError('Input Exclude Factor List Type Error')
    for factor in exclude_factor_list:
        if factor not in ['BETA', 'RESVOL', 'MOMENTUM', 'SIZE', 'SIZENL', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE',
                          'LIQUIDTY']:
            raise InputConfigError('Input Risk Factor Config Error')


def isnull(var):
    '''
    @summary: check the variable @var is None or empty array/timeseries/dataframe or not
    @var: input variable
    @return: bool
    '''
    if var is None:
        return True
    elif var is np.nan:
        return True
    elif isinstance(var, np.ndarray) and var.size == 0:
        return True
    elif isinstance(var, (pd.Series, pd.Index)) and var.size == 0:
        return True
    elif isinstance(var, pd.DataFrame) and (var.shape[0] == 0 or var.shape[1] == 0):
        return True
    else:
        return False


def handle_recv_json(requests_return):
    '''
    :param requests_return: JSON received from server
    :param flag:
    :param record_nan: the dict record Nan value
    :return:
    '''
    try:
        input_dict = requests_return.json()
    except:
        raise MFOutputError('JSON decoder error')

    if input_dict.get('result').get('status'):
        res_dict = input_dict.get('data')
        if not isinstance(res_dict, dict):
            res_dict = json.loads(input_dict.get('data'))
        new_dict = {}
        for k, v in res_dict.items():
            new_dict[str(k)] = v
        return new_dict
    else:
        raise MFOutputError(input_dict.get('result').get('error'))







