
# -*- coding: UTF-8 -*-
import os


from . import DATAYES

__author__ = 'frank'


from . import api_base
try:
    from StringIO import StringIO
except:
    from io import StringIO
import pandas as pd
import sys
import logging
import traceback

import requests

username = os.environ.get('DatayesPrincipalName', 'invalid')

def MktOptionTicksHistOneDayGet(optionId, date, field = "", startSecOffset = "", endSecOffset = "", pandas = "1"):

    fieldStr = ''
    if hasattr(field,'__iter__'):
        for temp in field:
            fieldStr += temp + ","
        fieldStr = fieldStr[:-1]
    else:
        fieldStr += field

    URL = '/api/market/getOptionTicksHistOneDay.csv?field=%s&optionId=%s&date=%s&startSecOffset=%s&endSecOffset=%s' % (fieldStr, optionId, date, startSecOffset, endSecOffset)

    httpClient = api_base.__getConn__()
    csvString = api_base.__getCSV__(URL, httpClient)
    if pandas != "1":
        return csvString
    if csvString is None or len(csvString) == 0 or csvString[0] == '-':
        raise Exception(u'%s for request: %s' % (csvString, URL))
    try:

        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'optionId':'str'}, )
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EcoDataProGet(indicID, beginDate = "", endDate = "", field = ""):
    """
    通联新宏观行业。输入指标ID，获取相关数据。

    :param indicID: 指标代码，可以通过特色数据页面获取。,可以是列表
    :param beginDate: 根据数据截止日期范围查询的开始日期，输入格式“YYYYMMDD”,可空
    :param endDate: 根据数据截止日期范围查询的结束日期，输入格式“YYYYMMDD”,可空
    :param field: 所需字段,可以是列表,可空；候选列表：indicID - 指标ID，publishDate - 数据源发布时间，periodDate - 数据期，dataValue - 数据值，updateTime - 更新时间
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
    if not api_base.is_pro_user() and not api_base.is_enterprise_user():
        raise Exception('Not uqer pro user!')
    api_name_info = DATAYES.EcoInfoProGet(indicID=indicID, field='indicID,dataApiName').drop_duplicates()
    df_list = []
    for idx, row in api_name_info.iterrows():
        thisIndicID, api_name = (row['indicID'], row['dataApiName'])
        f = getattr(DATAYES, api_name[3:] + 'Get')
        df = f(str(thisIndicID), beginDate, endDate, field)
        df_list.append(df)
    if df_list:
        return pd.concat(df_list)
    return pd.DataFrame()

if os.getenv('env', 'dev') != 'dev':
    cache_server = "tcp://db_cache_server:1111"
else:
    if os.environ.get('enterprise', '0') == '1':
        cache_server = "tcp://10.24.51.170:1111"  # stg
    else:
        cache_server = "tcp://10.22.132.36:1111"  # stg


batch_size = 5

def _get_service():
    import zerorpc
    client = zerorpc.Client(heartbeat=None, timeout=300)
    client.connect(cache_server)
    return client

def _load_factor(secID, fields, beginDate, endDate, username):
    try:
        client = _get_service()
        return 1, client.get_user_factors(secID, fields, beginDate, endDate, username)
    except Exception as e:
        error_msg = str(e)
        if "[Name Check]" in error_msg:
            raise Exception(error_msg[error_msg.find('[Name Check]')+12:])
        if '[Restriction]' in error_msg:
            raise Exception(error_msg[error_msg.find('[Restriction]')+13:])
        else:
            logging.error(traceback.format_exc().replace('\n', ''))
            raise Exception("The data is not available at this moment.")
    finally:
        client.close()

def FactorsGet(secID, beginDate, endDate, field):
    """
    获取因子数据，包括优矿因子、用户自定义因子和公共池因子
    :param secID: 股票池列表
    :param beginDate: 开始时间
    :param endDate: 结束时间
    :param field: 因子名列表，比如['PB', 'public.factor1', 'private.factor2']，不带前缀表示优矿因子，带private前缀表示用户自定义因子，带public前缀表示公共池因子
    :return: 一个pandas Panel，比如panel['PB']表示PB因子数据的DataFrame，panel['public.factor1']表示public.factor1因子数据的DataFrame，panel['private.factor2']表示private.factor2因子数据的DataFrame
    """
    from gevent.pool import Pool
    if not api_base.is_pro_user() and not api_base.is_enterprise_user():
        print ('非专业版用户，无法使用该API')
        return
    try:
        if type(field) != list:
            field = field.split(',')
    except:
        raise Exception('Invalid field format')
    client = _get_service()
    try:
        factors = client.translate_user_factors(field, username, beginDate)
    except Exception as e:
        error_msg = str(e)
        if "[Name Check]" in error_msg:
            print ("ERROR: " + error_msg[error_msg.find('[Name Check]')+12:])
            return
        if '[Restriction]' in error_msg:
            print ("ERROR: " + error_msg[error_msg.find('[Restriction]')+13:])
            return
        else:
            logging.error(traceback.format_exc().replace('\n', ''))
            raise Exception("The data is not available at this moment.")
    finally:
        client.close()
    result_factors = []
    from gevent.pool import Pool
    pool = Pool(batch_size)
    workers = []
    for key in factors:
        for f_id in factors[key]:
            factor = factors[key][f_id]
            result_factors.append(factor)

    group_factors = api_base.splist(result_factors, 3)
    for group in group_factors:
        worker = pool.spawn(_load_factor, secID, group, beginDate, endDate, username)
        workers.append(worker)

    result = {}
    if len(workers) > 0:
        pool.join()
        for worker in workers:
            val = worker.value
            if val[0] < 0:
                raise Exception(val[1])
            ast_data = val[1]
            for key in ast_data:
                result[key] = pd.DataFrame(ast_data[key])
    return pd.Panel(result)
