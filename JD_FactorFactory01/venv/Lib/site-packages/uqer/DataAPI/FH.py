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

__doc__="福汇"
def __FXTickRTSnapshotGet(currencyPair, field = "", pandas = "1"):
    """
    获取福汇外汇市场信息快照
    
    :param currencyPair: 货币对，包含以下几种：USD|CHF，EUR|USD，GBP|USD，GBP|AUD，USD|CAD，NZD|USD，EUR|JPY，NZD|CAD，AUD|NZD，EUR|GBP，GBP|CAD，GBP|JPY，AUD|JPY，USD|JPY，EUR|CHF，USD|CNH，EUR|AUD，AUD|CHF，AUD|USD，USD|HKD，不输入返回全部,可以是列表
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
    requestString.append('/api/foreignMarket/getFXTickRTSnapshot.csv?ispandas=1&') 
    requestString.append("currencyPair=")
    if hasattr(currencyPair,'__iter__') and not isinstance(currencyPair, str):
        if len(currencyPair) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = currencyPair
            requestString.append(None)
        else:
            requestString.append(','.join(currencyPair))
    else:
        requestString.append(currencyPair)
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
            api_base.handle_error(csvString, '__FXTickRTSnapshotGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__FXTickRTSnapshotGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'currencyPair', u'offerID', u'quoteID', u'tradeDate', u'dataTime', u'bid', u'ask', u'Low', u'High', u'volume', u'valueDate', u'buyInterest', u'sellInterest', u'contractCurrency', u'contractMultiplier', u'Digits', u'pointSize', u'tradingStatus', u'localTimestamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'currencyPair': 'str','quoteID': 'str','tradeDate': 'str','dataTime': 'str','valueDate': 'str','contractCurrency': 'str','tradingStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __FXTickRTIntraDayGet(currencyPair, field = "", pandas = "1"):
    """
    获取福汇外汇市场当日Tick数据信息
    
    :param currencyPair: 货币对，包含EUR|USD、USD|JPY、GBP|USD、USD|CHF、EUR|CHF、AUD|USD、AUD|CHF、USD|CAD、NZD|USD、EUR|GBP、USD|CNH、EUR|JPY、GBP|JPY、EUR|AUD、AUD|JPY、USD|HKD、GBP|CAD、GBP|AUD、NZD|CAD、AUD|NZD
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
    requestString.append('/api/foreignMarket/getFXTickRTIntraDay.csv?ispandas=1&') 
    if not isinstance(currencyPair, str) and not isinstance(currencyPair, unicode):
        currencyPair = str(currencyPair)

    requestString.append("currencyPair=%s"%(currencyPair))
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
            api_base.handle_error(csvString, '__FXTickRTIntraDayGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__FXTickRTIntraDayGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'updateDate', u'updateTime', u'offerID', u'currencyPair', u'quoteId', u'bid', u'ask', u'low', u'high', u'volume', u'sellInterest', u'buyInterest', u'contractCurrency', u'digits', u'pointSize', u'contractMultiplier', u'tradingStatus', u'valueDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'currencyPair': 'str','quoteId': 'str','contractCurrency': 'str','tradingStatus': 'str','valueDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __FXBarRTIntraDayGet(currencyPair, field = "", pandas = "1"):
    """
    高频数据，获取福汇外汇交易市场当日分钟线数据
    
    :param currencyPair: 货币对，包含EUR|USD、USD|JPY、GBP|USD、USD|CHF、EUR|CHF、AUD|USD、AUD|CHF、USD|CAD、NZD|USD、EUR|GBP、USD|CNH、EUR|JPY、GBP|JPY、EUR|AUD、AUD|JPY、USD|HKD、GBP|CAD、GBP|AUD、NZD|CAD、AUD|NZD,可以是列表
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
    requestString.append('/api/foreignMarket/getFXBarRTIntraDay.csv?ispandas=1&') 
    requestString.append("currencyPair=")
    if hasattr(currencyPair,'__iter__') and not isinstance(currencyPair, str):
        if len(currencyPair) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = currencyPair
            requestString.append(None)
        else:
            requestString.append(','.join(currencyPair))
    else:
        requestString.append(currencyPair)
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
            api_base.handle_error(csvString, '__FXBarRTIntraDayGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__FXBarRTIntraDayGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'dataDate', u'currencyPair', u'barTime', u'openBid', u'highBid', u'lowBid', u'closeBid', u'openAsk', u'highAsk', u'lowAsk', u'closeAsk', u'totalTicks']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'currencyPair': 'str','barTime': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __FXBarHistOneDayGet(currencyPair, tradeDate, field = "", pandas = "1"):
    """
    获取外汇交易市场历史某一日的分钟线数据
    
    :param currencyPair: 货币对，包含EUR|USD、USD|JPY、GBP|USD、USD|CHF、EUR|CHF、AUD|USD、AUD|CHF、USD|CAD、NZD|USD、EUR|GBP、USD|CNH、EUR|JPY、GBP|JPY、EUR|AUD、AUD|JPY、USD|HKD、GBP|CAD、GBP|AUD、NZD|CAD、AUD|NZD
    :param tradeDate: 交易日期，格式为YYYYMMDD
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
    requestString.append('/api/foreignMarket/getFXBarHistOneDay.csv?ispandas=1&') 
    if not isinstance(currencyPair, str) and not isinstance(currencyPair, unicode):
        currencyPair = str(currencyPair)

    requestString.append("currencyPair=%s"%(currencyPair))
    if not isinstance(tradeDate, str) and not isinstance(tradeDate, unicode):
        tradeDate = str(tradeDate)

    requestString.append("&tradeDate=%s"%(tradeDate))
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
            api_base.handle_error(csvString, '__FXBarHistOneDayGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__FXBarHistOneDayGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'currencyPair', u'tradeDate', u'barTime', u'openBid', u'highBid', u'lowBid', u'closeBid', u'openAsk', u'highAsk', u'lowAsk', u'closeAsk', u'totalTick']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'currencyPair': 'str','barTime': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __FXTickHistOneDayGet(currencyPair, tradeDate, field = "", pandas = "1"):
    """
    获取外汇交易市场历史某一日的Tick数据
    
    :param currencyPair: 货币对，包含EUR|USD、USD|JPY、GBP|USD、USD|CHF、EUR|CHF、AUD|USD、AUD|CHF、USD|CAD、NZD|USD、EUR|GBP、USD|CNH、EUR|JPY、GBP|JPY、EUR|AUD、AUD|JPY、USD|HKD、GBP|CAD、GBP|AUD、NZD|CAD、AUD|NZD
    :param tradeDate: 更新日期，经格林威治时间转为北京时间
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
    requestString.append('/api/foreignMarket/getFXTickHistOneDay.csv?ispandas=1&') 
    if not isinstance(currencyPair, str) and not isinstance(currencyPair, unicode):
        currencyPair = str(currencyPair)

    requestString.append("currencyPair=%s"%(currencyPair))
    if not isinstance(tradeDate, str) and not isinstance(tradeDate, unicode):
        tradeDate = str(tradeDate)

    requestString.append("&tradeDate=%s"%(tradeDate))
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
            api_base.handle_error(csvString, '__FXTickHistOneDayGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__FXTickHistOneDayGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'offerID', u'currencyPair', u'quoteID', u'bid', u'ask', u'Low', u'High', u'volume', u'updateDate', u'updateTime', u'sellInterest', u'buyInterest', u'contractCurrency', u'Digits', u'pointSize', u'contractMultiplier', u'tradingStatus', u'valueDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'currencyPair': 'str','quoteID': 'str','contractCurrency': 'str','tradingStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

