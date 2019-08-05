# -*- coding: UTF-8 -*-
from . import api_base
try:
    from StringIO import StringIO
except:
    from io import StringIO
import pandas as pd
import sys
from datetime import datetime
from .api_base import get_cache_key, get_data_from_cache, put_data_in_cache, pretty_traceback
import inspect
try:
    unicode
except:
    unicode = str

__doc__="IVolatility"
def DerIvGet(SecID, beginDate, endDate, optID = "", field = "", pandas = "1"):
    """
    原始隐含波动率,包括期权价格、累计成交量、持仓量、隐含波动率等。
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param optID: 期权合约编码,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIv.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&optID=")
    if hasattr(optID,'__iter__') and not isinstance(optID, str):
        if len(optID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = optID
            requestString.append(None)
        else:
            requestString.append(','.join(optID))
    else:
        requestString.append(optID)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'exchangeCD', u'secShortName', u'tradeDate', u'closePriceAdj', u'optID', u'expDate', u'strikePrice', u'contractType', u'sontractStyle', u'ask', u'bid', u'meanPrice', u'settlePrice', u'IV', u'turnoverVol', u'openInt', u'closePriceUnadj', u'forwardPrice', u'isInterPolated', u'delta', u'vega', u'gamma', u'theta', u'rho']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','optID': 'str','contractType': 'str','sontractStyle': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def DerIvHvGet(SecID, beginDate, endDate, period = "", field = "", pandas = "1"):
    """
    历史波动率，各个时间段的收盘－收盘历史波动率。
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param period: 期限,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIvHv.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&period=")
    if hasattr(period,'__iter__') and not isinstance(period, str):
        if len(period) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = period
            requestString.append(None)
        else:
            requestString.append(','.join(period))
    else:
        requestString.append(period)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvHvGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvHvGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'exchangeCD', u'tradeDate', u'closePriceUnadj', u'period', u'hv']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def DerIvIndexGet(SecID, beginDate, endDate, period = "", field = "", pandas = "1"):
    """
    隐含波动率指数，衡量30天至1080天到期平价期权的平均波动性的主要方法。
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param period: 期限,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIvIndex.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&period=")
    if hasattr(period,'__iter__') and not isinstance(period, str):
        if len(period) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = period
            requestString.append(None)
        else:
            requestString.append(','.join(period))
    else:
        requestString.append(period)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvIndexGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvIndexGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'exchangeCD', u'tradeDate', u'closePriceAdj', u'period', u'ivIndexCall', u'ivIndexPut', u'ivIndexMean']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def DerIvIvpDeltaGet(SecID, beginDate, endDate, period = "", delta = "", field = "", pandas = "1"):
    """
    隐含波动率曲面(基于参数平滑曲线)，基于delta（0.1至0.9,0.05升步）和到期日（1个月至3年）而标准化的曲面。
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param period: 期限,可以是列表,可空
    :param delta: 德尔塔,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIvIvpDelta.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&period=")
    if hasattr(period,'__iter__') and not isinstance(period, str):
        if len(period) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = period
            requestString.append(None)
        else:
            requestString.append(','.join(period))
    else:
        requestString.append(period)
    requestString.append("&delta=")
    if hasattr(delta,'__iter__') and not isinstance(delta, str):
        if len(delta) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = delta
            requestString.append(None)
        else:
            requestString.append(','.join(delta))
    else:
        requestString.append(delta)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvIvpDeltaGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvIvpDeltaGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'exchangeCD', u'tradeDate', u'closePriceUnadj', u'period', u'delta', u'moneyness', u'strikePrice', u'IV']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def DerIvParamGet(SecID, beginDate, endDate, expDate = "", field = "", pandas = "1"):
    """
    隐含波动率参数化曲面，由二阶方程波动曲线在每个到期日平滑后的曲面（a,b,c曲线系数）
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param expDate: 到期日期,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIvParam.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&expDate=")
    if hasattr(expDate,'__iter__') and not isinstance(expDate, str):
        if len(expDate) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = expDate
            requestString.append(None)
        else:
            requestString.append(','.join(expDate))
    else:
        requestString.append(expDate)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvParamGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvParamGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'exchangeCD', u'expDate', u'tradeDate', u'closePriceUnadj', u'forwardPrice', u'parabolaA', u'parabolaB', u'parabolaC']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def DerIvRawDeltaGet(SecID, beginDate, endDate, period = "", delta = "", field = "", pandas = "1"):
    """
    隐含波动率曲面(基于原始隐含波动率)，基于delta（0.1至0.9,0.05升步）和到期日（1个月至3年）而标准化的曲面。
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param period: 期限,可以是列表,可空
    :param delta: 德尔塔,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIvRawDelta.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&period=")
    if hasattr(period,'__iter__') and not isinstance(period, str):
        if len(period) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = period
            requestString.append(None)
        else:
            requestString.append(','.join(period))
    else:
        requestString.append(period)
    requestString.append("&delta=")
    if hasattr(delta,'__iter__') and not isinstance(delta, str):
        if len(delta) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = delta
            requestString.append(None)
        else:
            requestString.append(','.join(delta))
    else:
        requestString.append(delta)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvRawDeltaGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvRawDeltaGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'tradeDate', u'closePriceUnadj', u'period', u'delta', u'moneyness', u'strikePrice', u'IV']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def DerIvSurfaceGet(SecID, beginDate, endDate, contractType = "", field = "", pandas = "1"):
    """
    隐含波动率曲面(在值程度)，基于在值程度而标准化的曲面。执行价格区间在-60%到+60%，5%升步，到期区间为1个月至3年。
    
    :param SecID: 证券展示代码
    :param beginDate: 起始日期，输入格式“YYYYMMDD”
    :param endDate: 截止日期，输入格式“YYYYMMDD”
    :param contractType: 期权类型,C代表认购，P代表认沽,可以是列表,可空
    :param field: 所需字段,可以是列表,可空
    :param pandas: 1表示返回 pandas data frame，0表示返回csv,可空
    :return: :raise e: API查询的结果，是CSV或者被转成pandas data frame；若查询API失败，返回空data frame； 若解析失败，则抛出异常
    """
        
    pretty_traceback()
    frame = inspect.currentframe()
    func_name, cache_key = get_cache_key(frame)
    cache_result = get_data_from_cache(func_name, cache_key)
    if cache_result is not None:
        return cache_result
    split_index = None
    split_param = None
    httpClient = api_base.__getConn__()    
    requestString = []
    requestString.append('/api/IV/getDerIvSurface.csv?ispandas=1&') 
    if not isinstance(SecID, str) and not isinstance(SecID, unicode):
        SecID = str(SecID)

    requestString.append("SecID=%s"%(SecID))
    try:
        beginDate = beginDate.strftime('%Y%m%d')
    except:
        beginDate = beginDate.replace('-', '')
    requestString.append("&beginDate=%s"%(beginDate))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    requestString.append("&contractType=")
    if hasattr(contractType,'__iter__') and not isinstance(contractType, str):
        if len(contractType) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = contractType
            requestString.append(None)
        else:
            requestString.append(','.join(contractType))
    else:
        requestString.append(contractType)
    requestString.append("&field=")
    if hasattr(field,'__iter__') and not isinstance(field, str):
        if len(field) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = field
            requestString.append(None)
        else:
            requestString.append(','.join(field))
    else:
        requestString.append(field)
    if split_param is None:
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'DerIvSurfaceGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'DerIvSurfaceGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'SecID', u'ticker', u'tradeDate', u'closePriceUnadj', u'period', u'strike', u'outOfMoney', u'contractType', u'IV', u'delta']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'SecID': 'str','ticker': 'str','contractType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

