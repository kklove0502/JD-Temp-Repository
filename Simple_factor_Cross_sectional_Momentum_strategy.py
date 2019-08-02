# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')

"""
数据处理部分
"""

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

    df = pd.read_csv('C:/Software/Pycharm/JD_TEST/data/%s' % i, names=['Date','Open','High','Low','Close','Volume','OpenInterest'])

    # 正则提取品种名称作为一列
    pattern = re.compile(r'[A-Za-z]')
    symbol_str = ''.join(pattern.findall(i, 0, 2))
    df['Symbol'] = symbol_str

    # 转换时间序列
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index(['Date','Symbol'], inplace=True)

    return df

def cal_return(df):

    df = df[['Close']]
    df['Return'] = df['Close'].pct_change()
    df['NextReturn'] = df['Return'].shift(-1)

    return df

def cal_CumReturn(df, R):
    '''
    为指定的df计算rolling=R的累积收益
    :param df：dataframe，对应的df
    :param R: int，rolling期限
    :return:
    '''
    df['CumReturn_%d' % R] = (df['Return']+1).rolling(R).apply(np.prod) - 1
    df['CumReturn_%d' % R] = df['CumReturn_%d' % R].shift(1)
    return df

def concat_func(df_list):
    """
    拼接功能函数，按行进行合并(按行，就是添加行的意思、即竖着往下加一行再加一行)
    :param corr_df_list: list里面每个元素是一个df
    :return:
    """
    concat = lambda x,y: pd.concat((x,y), axis=0)
    factor_df = reduce(concat, df_list)
    return factor_df

def handle_data_momentum_factor1(R, path):
    futures_df_list = []
    pathDir = os.listdir(path)
    for i in pathDir:
        df = csv_to_df(i)
        df = cal_return(df)
        df = cal_CumReturn(df,R)
        df = df.dropna()
        futures_df_list.append(df)

    return concat_func(futures_df_list)

big_df = handle_data_momentum_factor1(R=10, path='C:/Software/Pycharm/JD_TEST/data')

big_df.sort_index(level='Date', inplace=True)

"""
策略逻辑部分
"""

def cal_nav(big_df):
    '''

    :param big_df:
    :return:
    '''
    date_list = []
    nav_list = []

    dates = get.get_dates_list(big_df)

    for date in dates:

        df = big_df.loc[date]
        df['Rank'] = df['CumReturn_10'].rank()
        max, min = df['Rank'].max(), df['Rank'].min()
        df['Position'] = np.where(df['Rank'] == max, -1, 0)
        df['Position'] = np.where(df['Rank'] == min, 1, df['Position'])

        ####用前一天的收盘数据算出信号，得到今天计划开空仓、开多仓的品种标的，昨收价开仓、今收价平仓####
        ####手续费单边万分之三####
        df['Strategy'] = df['Position'] * df['Return']
        nav = (1 + 0.5 * df['Strategy'].sum()) * 0.9994
        date_list.append(date)
        nav_list.append(nav)

    nav_sr = pd.Series(nav_list, index=date_list)

    nav_daily_sr = nav_sr.sort_index()
    nav_sr = nav_daily_sr.cumprod()

    return nav_daily_sr, nav_sr

nav_daily_sr, nav_sr = cal_nav(big_df)
nav_df = pd.DataFrame(list(zip(nav_daily_sr, nav_sr)), columns=['nav_daily_sr', 'nav_sr'], index=nav_daily_sr.index)

EAR = (1 + nav_sr[-1]) ** (250 / nav_sr.index.__len__()) - 1
print('策略年化利率为：%.2f%%' % (EAR * 100))


"""
画图部分
"""

plt.figure(figsize=(8,6))
plt.subplot(211)
plt.plot(nav_df['nav_sr'], lw=1.5, label='nav')
plt.legend(loc=0)
plt.ylabel('nav')
plt.title('Plot')

plt.subplot(212)
plt.plot(nav_df['nav_daily_sr'], 'g', lw=1.5, label='nav_daily')
plt.legend(loc=0)
plt.xlabel('t')
plt.ylabel('nav_daily')
plt.show()

"""
IC指标计算
"""

def computeCorrelation(x,y):
    """
    功能函数，计算简单线性回归相关系数的函数
    :param x,y: series，计算相关系数的两个序列
    :return: r：float，相关系数
    """
    import math
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0,len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar**2
        varY += difYYbar**2
    SST = math.sqrt(varX * varY)
    return SSR/SST

def normal_ic_caculator(df):
    """
    计算normal_ic，某时点某因子在全部股票上的暴露值与下一期回报(个股收益率)的截面相关系数
    :param df: dataframe，用Handle_df_proc5处理好的dataframe
    :return: normal_ic_sr：series，index是date、value是corr因子的normal_ic值
    """
    normal_ic_dates_list = []
    normal_ic_values_list = []
    dates = get.get_dates_list(df)

    for date in dates:
        value = computeCorrelation(((df.loc[date])['CumReturn_10']),((df.loc[date])['NextReturn']))
        normal_ic_dates_list.append(date)
        normal_ic_values_list.append(value)

    normal_ic_sr = pd.Series(normal_ic_values_list, index=normal_ic_dates_list)
    normal_ic_sr = normal_ic_sr.sort_index()
    return normal_ic_sr

normal_ic_sr = normal_ic_caculator(big_df)

normal_ic_df = pd.DataFrame(normal_ic_sr, index=normal_ic_sr.index, columns=['IC'])

# IC方向分析
ic_mean = normal_ic_sr.mean()
normal_ic_df['Direction'] = np.where(normal_ic_df['IC'] > 0, 1, -1)
grouped = normal_ic_df.groupby(normal_ic_df['Direction'])
grouped.count()

"""
画图
"""
plt.figure(figsize=(8, 6))
plt.plot(normal_ic_sr, lw=1.5)

plt.grid(True)
plt.xlabel('time')
plt.ylabel('Value')
plt.title('Normal IC')
plt.show()