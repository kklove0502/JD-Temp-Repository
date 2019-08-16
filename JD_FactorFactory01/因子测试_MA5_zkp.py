"""
横截面因子，用品种指数数据XX000加工信号；

"""

import pandas as pd
import os
import re
import numpy as np
from functools import reduce
#from tools.compute import computeCorrelation
import warnings; warnings.simplefilter('ignore')

def computeCorrelation(x,y):
    """
    计算功能函数，计算简单线性回归相关系数的函数
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

    df = pd.read_csv('D:/PythonProject/JD_FactorFactory01/DataDownload/day/%s' % i, names=['Date','Open','High','Low','Close','Volume','OpenInterest'])

    # 正则提取品种名称作为一列
    pattern = re.compile(r'[A-Za-z]')
    symbol_str = ''.join(pattern.findall(i, 0, 2))
    df['Symbol'] = symbol_str

    # 转换时间序列
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index(['Date','Symbol'], inplace=True)
    #print(df)

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
'''
def cal_CR(df):
    """
    计算因子值CR相关的column
    :param df:
    :return:
    """
    df['CR'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']) * np.sqrt(df['Volume'])

    return df
'''
def MA(c,n):
    z = []
    for i in range(0, len(c), 1):
        if i<n :
            b=0
        else :
            b=i-n+1
                
        temp=c[b:i+1]
        av=np.mean(temp)
        z.append(av)
    return z

def cal_CR(df):
    """
    计算因子值CR相关的column
    :param df:
    :return:
    """
    close=df['Close'].values
    #print(len(close))
    mavalue=MA(close,5)
    #print(mavalue)
    df['CR'] = mavalue

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

def handle_data_CR_factor(path):
    """
    ☆☆核心函数☆☆
    把所有时间频率相同的TB的csv数据放在一个文件夹目录下，输入path，直接加工成需要的大df；
    在函数里面不断添加新的cal函数，计算更多的column（即因子），如cal_CR等等；
    :param path:
    :return:
    """
    futures_df_list = []
    pathDir = os.listdir(path)
    
    for i in pathDir:
        df = csv_to_df(i)
        df = cal_Return(df)
        df = cal_CR(df)
        df = df.dropna()
        futures_df_list.append(df)

    return concat_func(futures_df_list)

def hadle_data_for_JQ(df,factor):
    """
    把因子数据转换成可以在JQ中进行回测的格式；
    因子IC检测部分用不到此函数；
    :param df:
    :param factor:
    :return:
    """
    df = df.reset_index()
    df = df[['Date','Symbol',factor]]
    df = df.pivot(index='Date', columns='Symbol', values=factor)
    return df

def indicator_CS_IC(df, factor, beginDate, endDate):
    """
    横截面IC指标计算函数
    :param df: dataframe，包含因子值
    :return: CS_IC_sr: series, 横截面IC指标序列、index是日期value是IC值
    ？？有个问题，就是输入的beginDate/endDate如果不是交易日，取不出来数据，怎么解决？？
    先人工判断好交易日历
    """
    CS_IC_date_list = []
    CS_IC_value_list = []
    # 这里要求df的第一层索引是Date、第二层索引是Symbol；如果第二层索引是str即symbol、pass，否则swap
    if isinstance((df.index.get_level_values(level=1))[0], str):
        pass
    else:
        df = df.swaplevel()
    # 因为get方法会使index乱序，这里重新sort一下
    df.sort_index(level=0, inplace=True)
    # df切片选出符合时间区间的行
    df = df.loc[beginDate:endDate]
    # 循环计算IC
    for date in get.get_dates_list(df):
        factor_sr = (df.loc[date])[factor]
        #print(str(factor_sr)+'|'+str(date))
        NextReturn_sr = (df.loc[date])['NextReturn']
        value = factor_sr.corr(NextReturn_sr,method='spearman')
        CS_IC_date_list.append(date)
        CS_IC_value_list.append(value)

    CS_IC_sr = pd.Series(CS_IC_value_list, index=CS_IC_date_list)
    CS_IC_sr.sort_index(inplace=True)
    

    return CS_IC_sr


if __name__ == '__main__':

    df = handle_data_CR_factor('D:/PythonProject/JD_FactorFactory01/DataDownload/day')
    test = indicator_CS_IC(df, 'CR', beginDate='20100105', endDate='20181227')
    
    # test.plot(kind='line')


    # 数据导出，导出给JQ平台做因子排序用的csv文件，和因子检测无关
    # df_CR = hadle_data_for_JQ(df,'CR')
    # df_CR.to_csv('D:/PythonProject/JD_FactorFactory01/DataDownload/factor_for_JQ/df_CR.csv')