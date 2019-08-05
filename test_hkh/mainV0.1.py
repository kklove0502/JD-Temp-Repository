# -*- coding: utf-8 -*-
# coding: utf-8

import pandas as pd
import os
import re
import numpy as np
from functools import reduce
from tools.compute import computeCorrelation

# from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')

def concat_func(df_list):
    """
    拼接功能函数，按行进行合并(按行，就是添加行的意思、即竖着往下加一行再加一行)
    :param corr_df_list: list里面每个元素是一个df
    :return:
    """
    concat = lambda x,y: pd.concat((x,y), axis=0)
    factor_df = reduce(concat, df_list)
    return factor_df

class get():
    """
    输入multi_index_dataframe，index是date和code，几个获取日期或者代码的方法
    ps.按照factor_df格式已设置默认level；level0是dates，level1是codes
    """
    def get_first_day(df,level=0):
        """
        取factor_df中的第一天
        :param level: int, 默认=0
        :return: date：timestamp
        """
        return min(list(set(df.index.get_level_values(level))))

    def get_last_day(df,level=0):
        return max(list(set(df.index.get_level_values(level))))

    def get_dates_list(df,level=0):
        return list(set(df.index.get_level_values(level)))

    def get_stocks_list(df,level=1):
        return list(set(df.index.get_level_values(level)))

def csv_to_df(i):
    '''
    把单个csv读取进来并整理成需要的单个df
    :param i: str，文件名，形如'CJ888.csv'
    :return: df，multi_index为Date，Symbol
    '''

    df = pd.read_csv('C:/Software/Pycharm/test_hkh/DataDownload/%s' % i, names=['Date','Open','High','Low','Close','Volume','OpenInterest'], encoding='gbk')

    # 正则提取品种名称作为一列
    pattern = re.compile(r'[A-Za-z]')
    symbol_str = ''.join(pattern.findall(i, 0, 2))
    df['Symbol'] = symbol_str

    # 转换时间序列
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index(['Date','Symbol'], inplace=True)

    return df

def cal_Return(df):
    """
    计算收益率Return相关的columns
    :param df:
    :return:
    """
    df['Return'] = df['Close'].pct_change()
    df['NextReturn'] = df['Return'].shift(-1)
    df = df.dropna()

    return df

def cal_CR(df):
    """
    计算因子值CR相关的column
    :param df:
    :return:
    """
    df['CR'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']) * np.sqrt(df['Volume'])

    return df

# 不同的数据频率，存放在不同的文件目录path下面
def handle_data_CR_factor(path):
    futures_df_list = []
    pathDir = os.listdir(path)
    for i in pathDir:
        df = csv_to_df(i)
        df = cal_Return(df)
        df = cal_CR(df)
        df = df.dropna()
        futures_df_list.append(df)

    return concat_func(futures_df_list)

# 这里所有品种的起止时间是相同的，还需要考虑如果起止时间参差不齐的做法
def indicator_IC(df,factor,beginDate,endDate):
    """
    计算单个品种时间序列的IC指标；
    x序列是某品种时间序列的因子值，y序列是某品种时间序列的下一期收益率
    :param df:dataframe，multiIndex第一层是时间Date第二层是期货品种Symbol
    :param factor:str，因子名称，示例为'CR'
    :param beginDate:开始时间
    :param endDate:结束时间
    :return:series，index为品种名称、value为这个品种的IC指标
    """

    normal_IC_futures_list = []
    normal_IC_values_list = []
    futures = get.get_stocks_list(df)

    # 因为只能通过loc筛选第一层索引，需要通过if判断期货品种合约是否在第一层索引上，如果在第二层索引上，swaplevel()
    if isinstance((df.index.get_level_values(level=0))[0], str):
        pass
    else:
        df = df.swaplevel()

    # # query方法选出来的还是multiIndex
    # df_test.query('Symbol=="cu"')

    for future in futures:

        df_loc_by_futures = df.loc[future]
        df_loc_by_futures_dates = df_loc_by_futures.loc[beginDate:endDate]
        value = computeCorrelation(df_loc_by_futures_dates[factor],df_loc_by_futures_dates['NextReturn'])

        normal_IC_futures_list.append(future)
        normal_IC_values_list.append(value)

    IC_sr = pd.Series(normal_IC_values_list, index=normal_IC_futures_list)

    return IC_sr

if __name__ == '__main__':
    df = handle_data_CR_factor('C:/Software/Pycharm/test_hkh/DataDownload')
    IC_sr = indicator_IC(df, factor='CR', beginDate='20100101', endDate='20121231')
    print(IC_sr)

