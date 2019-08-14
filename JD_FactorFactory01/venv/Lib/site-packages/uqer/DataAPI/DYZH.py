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

__doc__="中汉指数"
def MktIdxdZHGet(tradeDate = "", beginDate = "", endDate = "", secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取中汉指数日度行情数据，包含指数开高低收价，收益率，超额收益等
    
    :param tradeDate: 交易日起，格式为yyyymmdd,tradeDate、beginDate、endDate至少选择一个
    :param beginDate: 交易日起，格式为yyyymmdd,tradeDate、beginDate、endDate至少选择一个
    :param endDate: 交易日起，格式为yyyymmdd,tradeDate、beginDate、endDate至少选择一个
    :param secID: 通联编制的证券编码，可传入证券交易代码使用DataAPI.SecIDGet接口获取到。,可以是列表,可空
    :param ticker: 交易代码,可以是列表,可空
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
    requestString.append('/api/idxMarket/getMktIdxdZH.csv?ispandas=1&') 
    try:
        tradeDate = tradeDate.strftime('%Y%m%d')
    except:
        tradeDate = tradeDate.replace('-', '')
    requestString.append("tradeDate=%s"%(tradeDate))
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
    requestString.append("&secID=")
    if hasattr(secID,'__iter__') and not isinstance(secID, str):
        if len(secID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secID
            requestString.append(None)
        else:
            requestString.append(','.join(secID))
    else:
        requestString.append(secID)
    requestString.append("&ticker=")
    if hasattr(ticker,'__iter__') and not isinstance(ticker, str):
        if len(ticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ticker
            requestString.append(None)
        else:
            requestString.append(','.join(ticker))
    else:
        requestString.append(ticker)
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=False)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'MktIdxdZHGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=False)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIdxdZHGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'tradeDate', u'secShortNameEn', u'secShortNameEn', u'currencyCD', u'indexLevel', u'comGroup', u'comGroupEn', u'comGroupCD', u'comSubGroup', u'comSubGroupEN', u'comSubGroupCD', u'singleCom', u'singleComEn', u'singleComCD', u'LS', u'LSen', u'leverage', u'weightMetho', u'weightMethoEn', u'indexValueER', u'dailyIndexER', u'indexValueTR', u'dailyIndexTR', u'openIndex', u'highIndex', u'lowIndex', u'closeIndex', u'wtdIndexER', u'wtdIndexTR', u'mtdIndexER', u'mtdIndexTR', u'ytdIndexER', u'ytdIndexTR', u'indexOpenInt', u'turnoverVol', u'indexValueERUSD', u'dailyIndexERUSD', u'dailyIndexTRUSD', u'indexValueEREUR', u'dailyIndexEREUR', u'dailyIndexTREUR', u'indexValueERJPY', u'dailyIndexERJPY', u'dailyIndexTRJPY', u'indexValueERHKD', u'dailyIndexERHKD', u'dailyIndexTRHKD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortNameEn': 'str','secShortNameEn': 'str','currencyCD': 'str','indexLevel': 'str','comGroup': 'str','comGroupEn': 'str','comGroupCD': 'str','comSubGroup': 'str','comSubGroupEN': 'str','comSubGroupCD': 'str','singleCom': 'str','singleComEn': 'str','singleComCD': 'str','LS': 'str','LSen': 'str','leverage': 'str','weightMetho': 'str','weightMethoEn': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIdxdZHAnalyGet(tradeDate = "", beginDate = "", endDate = "", secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取中汉指数日度分析数据，包含年化收益率，最大回撤，收益率标准差，夏普比率，索提诺比率等
    
    :param tradeDate: 交易日起，格式为yyyymmdd,tradeDate、beginDate、endDate至少选择一个
    :param beginDate: 交易日起，格式为yyyymmdd,tradeDate、beginDate、endDate至少选择一个
    :param endDate: 交易日起，格式为yyyymmdd,tradeDate、beginDate、endDate至少选择一个
    :param secID: 通联编制的证券编码，可传入证券交易代码使用DataAPI.SecIDGet接口获取到。,可以是列表,可空
    :param ticker: 交易代码,可以是列表,可空
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
    requestString.append('/api/idxMarket/getMktIdxdZHAnaly.csv?ispandas=1&') 
    try:
        tradeDate = tradeDate.strftime('%Y%m%d')
    except:
        tradeDate = tradeDate.replace('-', '')
    requestString.append("tradeDate=%s"%(tradeDate))
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
    requestString.append("&secID=")
    if hasattr(secID,'__iter__') and not isinstance(secID, str):
        if len(secID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secID
            requestString.append(None)
        else:
            requestString.append(','.join(secID))
    else:
        requestString.append(secID)
    requestString.append("&ticker=")
    if hasattr(ticker,'__iter__') and not isinstance(ticker, str):
        if len(ticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ticker
            requestString.append(None)
        else:
            requestString.append(','.join(ticker))
    else:
        requestString.append(ticker)
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=False)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'MktIdxdZHAnalyGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=False)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIdxdZHAnalyGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'tradeDate', u'secShortNameEn', u'secShortNameEn', u'currencyCD', u'indexLevel', u'comGroup', u'comGroupEn', u'comGroupCD', u'comSubGroup', u'comSubGroupEN', u'comSubGroupCD', u'singleCom', u'singleComEn', u'singleComCD', u'LS', u'LSen', u'leverage', u'weightMetho', u'weightMethoEn', u'indexER20d', u'indexTR20d', u'annuER20d', u'meanER20d', u'sd20d', u'maxDrawdown20d', u'annuShareRatioER20d', u'annuShareRatioTR20d', u'annuSortinoRatioER20d', u'annuSortinoRatioER20d', u'skewness20d', u'excessKurtosis20d']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortNameEn': 'str','secShortNameEn': 'str','currencyCD': 'str','indexLevel': 'str','comGroup': 'str','comGroupEn': 'str','comGroupCD': 'str','comSubGroup': 'str','comSubGroupEN': 'str','comSubGroupCD': 'str','singleCom': 'str','singleComEn': 'str','singleComCD': 'str','LS': 'str','LSen': 'str','leverage': 'str','weightMetho': 'str','weightMethoEn': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

