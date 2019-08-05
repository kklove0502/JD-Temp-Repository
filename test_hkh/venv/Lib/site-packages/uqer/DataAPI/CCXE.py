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

__doc__="中诚信资讯"
def MktEqudCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取交易所股票日行情，包含证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,昨收盘价,开盘价,最高价,最低价,收盘价,成交数量,成交金额,成交笔数等，历史追溯至1990年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'000001'，可获取到secID'000001.XSHE'后，在此输入'000001.XSHE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/equity/getMktEqudCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktEqudCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktEqudCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol', u'turnoverValue', u'dealAmount']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquCCXEGet(secID = "", ticker = "", exchangeCD = "", listSectorCD = "", equTypeCD = "", listStatusCD = "", field = "", pandas = "1"):
    """
    获取股票的基本信息，包含股票交易代码及其简称、股票类型、上市状态、上市板块、上市日期等；上市状态为最新数据，不显示历史变动信息。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,可空
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,可空
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,可空
    :param listSectorCD: 上市板块。例如，1-主板；2-创业板；3-中小板。对应DataAPI.SysCodeGet.codeTypeID=10016。,可以是列表,可空
    :param equTypeCD: 股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
    :param listStatusCD: 上市状态。L-上市；S-暂停；DE-终止上市；UN-未上市。对应DataAPI.SysCodeGet.codeTypeID=10005。,可以是列表,可空
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
    requestString.append('/api/equity/getEquCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&listSectorCD=")
    if hasattr(listSectorCD,'__iter__') and not isinstance(listSectorCD, str):
        if len(listSectorCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = listSectorCD
            requestString.append(None)
        else:
            requestString.append(','.join(listSectorCD))
    else:
        requestString.append(listSectorCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
    requestString.append("&listStatusCD=")
    if hasattr(listStatusCD,'__iter__') and not isinstance(listStatusCD, str):
        if len(listStatusCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = listStatusCD
            requestString.append(None)
        else:
            requestString.append(','.join(listStatusCD))
    else:
        requestString.append(listStatusCD)
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
            api_base.handle_error(csvString, 'EquCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'equTypeCD', u'listSectorCD', u'cnSpell', u'secShortName', u'secShortNameEn', u'secFullName', u'secFullNameEn', u'partyID', u'listDate', u'listStatusCD', u'delistDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','equTypeCD': 'str','listSectorCD': 'str','cnSpell': 'str','secShortName': 'str','secShortNameEn': 'str','secFullName': 'str','secFullNameEn': 'str','listStatusCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquIPOaCCXEGet(secID = "", exchangeCD = "", ticker = "", listDateStart = "", listDateEnd = "", field = "", pandas = "1"):
    """
    获取A股首次公开发行上市的基本信息，包含股票首次公开发行的进程及发行结果。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、exchangeCD、ticker至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、exchangeCD、ticker至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、exchangeCD、ticker至少选择一个
    :param listDateStart: 该期间所有的IPO股票上市日期，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有IPO的基本信息。,可空
    :param listDateEnd: 该期间所有的IPO股票上市日期，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有IPO的基本信息。,可空
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
    requestString.append('/api/equity/getEquIPOaCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    if not isinstance(listDateStart, str) and not isinstance(listDateStart, unicode):
        listDateStart = str(listDateStart)

    requestString.append("&listDateStart=%s"%(listDateStart))
    if not isinstance(listDateEnd, str) and not isinstance(listDateEnd, unicode):
        listDateEnd = str(listDateEnd)

    requestString.append("&listDateEnd=%s"%(listDateEnd))
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
            api_base.handle_error(csvString, 'EquIPOaCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquIPOaCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'intPublishDate', u'proPublishDate', u'listPublishDate', u'listDate', u'parValue', u'issuePrice', u'currencyCD', u'issueShares', u'issueRaiseCap', u'issueCost', u'issueCostPerShare', u'onlineIssueBeginDate', u'onlineIssueEndDate', u'offlineIssueBeginDate', u'offlineIssueEndDate', u'onlineIssueApplyCode', u'onlineIssueApplyAbbr', u'offlineIssueApplyCode', u'offlineIssueApplyAbbr', u'onlineIssueLottoRatio', u'onlineIssueShares', u'offlineIssueShares', u'issuePED', u'issuePEW', u'issueMode', u'sponsor', u'leadUnderwriter', u'coleadUnderwriter']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','intPublishDate': 'str','proPublishDate': 'str','listPublishDate': 'str','listDate': 'str','currencyCD': 'str','onlineIssueBeginDate': 'str','onlineIssueEndDate': 'str','offlineIssueBeginDate': 'str','offlineIssueEndDate': 'str','onlineIssueApplyCode': 'str','onlineIssueApplyAbbr': 'str','offlineIssueApplyCode': 'str','offlineIssueApplyAbbr': 'str','issueMode': 'str','sponsor': 'str','leadUnderwriter': 'str','coleadUnderwriter': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquIPObCCXEGet(secID = "", ticker = "", exchangeCD = "", listDateStart = "", listDateEnd = "", field = "", pandas = "1"):
    """
    获取B股首次公开发行上市的基本信息，包含股票首次公开发行的进程及发行结果。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param listDateStart: 该期间所有的IPO股票上市日期，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有IPO的基本信息。,可空
    :param listDateEnd: 该期间所有的IPO股票上市日期，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有IPO的基本信息。,可空
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
    requestString.append('/api/equity/getEquIPObCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    if not isinstance(listDateStart, str) and not isinstance(listDateStart, unicode):
        listDateStart = str(listDateStart)

    requestString.append("&listDateStart=%s"%(listDateStart))
    if not isinstance(listDateEnd, str) and not isinstance(listDateEnd, unicode):
        listDateEnd = str(listDateEnd)

    requestString.append("&listDateEnd=%s"%(listDateEnd))
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
            api_base.handle_error(csvString, 'EquIPObCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquIPObCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'proPublishDate', u'listPublishDate', u'listDate', u'parValue', u'issuePrice', u'frIssuePrice', u'frCurrencyCD', u'issueShares', u'issueRaiseCap', u'frIssueRaiseCap', u'issueCost', u'issueCostPerShare', u'frIssueCost', u'frIssueCostPerShare', u'issuePED', u'issuePEW', u'issueMode', u'underwritingModeCD', u'sponsor', u'leadUnderwriter', u'interCoord']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','proPublishDate': 'str','listPublishDate': 'str','listDate': 'str','frCurrencyCD': 'str','issueMode': 'str','underwritingModeCD': 'str','sponsor': 'str','leadUnderwriter': 'str','interCoord': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquDiviCCXEGet(secID = "", ticker = "", exchangeCD = "", equTypeCD = "", endDateStart = "", endDateEnd = "", field = "", pandas = "1"):
    """
    获取股票历次分红(派现、送股、转增股)的实施信息，包含股改分红情况。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param equTypeCD: 分红股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
    :param endDateStart: 本次分红所属的财政年度、财政半年度、财政季度的最后一个自然日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有分红,可空
    :param endDateEnd: 本次分红所属的财政年度、财政半年度、财政季度的最后一个自然日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有分红,可空
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
    requestString.append('/api/equity/getEquDiviCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
    if not isinstance(endDateStart, str) and not isinstance(endDateStart, unicode):
        endDateStart = str(endDateStart)

    requestString.append("&endDateStart=%s"%(endDateStart))
    if not isinstance(endDateEnd, str) and not isinstance(endDateEnd, unicode):
        endDateEnd = str(endDateEnd)

    requestString.append("&endDateEnd=%s"%(endDateEnd))
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
            api_base.handle_error(csvString, 'EquDiviCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquDiviCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'endDate', u'equTypeCD', u'imPublishDate', u'perCashDiv', u'perCashDivAfTax', u'currencyCD', u'frPerCashDiv', u'frPerCashDivAfTax', u'perShareDivRatio', u'perShareTransRatio', u'divObject', u'recordDate', u'exDivDate', u'bLastTradeDate', u'payCashDate', u'bonusShareListDate', u'totalCashDiv', u'baseShares', u'sharesAfDiv']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','endDate': 'str','equTypeCD': 'str','currencyCD': 'str','divObject': 'str','recordDate': 'str','exDivDate': 'str','bLastTradeDate': 'str','payCashDate': 'str','bonusShareListDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquDivpCCXEGet(secID = "", ticker = "", exchangeCD = "", equTypeCD = "", isPass = "", isDiv = "", field = "", pandas = "1"):
    """
    获取股票历次分红预案(派现、送股、转增股)的基本信息，包含历次分红预案的内容以及是否最终实施。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param equTypeCD: 分红股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
    :param isPass: 预案是否通过。1-通过；0-未通过。,可以是列表,可空
    :param isDiv: 是否分红。例如，0-否；05-分红。对应DataAPI.SysCodeGet.codeTypeID=20011。,可以是列表,可空
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
    requestString.append('/api/equity/getEquDivpCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
    requestString.append("&isPass=")
    if hasattr(isPass,'__iter__') and not isinstance(isPass, str):
        if len(isPass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isPass
            requestString.append(None)
        else:
            requestString.append(','.join(isPass))
    else:
        requestString.append(isPass)
    requestString.append("&isDiv=")
    if hasattr(isDiv,'__iter__') and not isinstance(isDiv, str):
        if len(isDiv) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isDiv
            requestString.append(None)
        else:
            requestString.append(','.join(isDiv))
    else:
        requestString.append(isDiv)
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
            api_base.handle_error(csvString, 'EquDivpCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquDivpCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'endDate', u'equTypeCD', u'isDiv', u'planPublishDate', u'isPass']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','endDate': 'str','equTypeCD': 'str','isDiv': 'str','planPublishDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquAllotpCCXEGet(secID = "", ticker = "", exchangeCD = "", equTypeCD = "", field = "", pandas = "1"):
    """
    获取公司历次配股预案及变动情况。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param equTypeCD: 发行股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
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
    requestString.append('/api/equity/getEquAllotpCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
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
            api_base.handle_error(csvString, 'EquAllotpCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquAllotpCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'equTypeCD', u'planPublishDate', u'vldBeginDate', u'vldEndDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','equTypeCD': 'str','planPublishDate': 'str','vldBeginDate': 'str','vldEndDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquAllotiCCXEGet(secID = "", ticker = "", exchangeCD = "", equTypeCD = "", field = "", pandas = "1"):
    """
    获取公司历次配股的实施信息，包括配股价、配股比例、配股数量等信息。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param equTypeCD: 发行股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
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
    requestString.append('/api/equity/getEquAllotiCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
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
            api_base.handle_error(csvString, 'EquAllotiCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquAllotiCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'allotYear', u'equTypeCD', u'proPublishDate', u'allotCode', u'allotAbbr', u'allotmentPrice', u'allotFrPrice', u'currencyCD', u'allotmentRatio', u'baseShares', u'allotShares', u'recordDate', u'exRightsDate', u'payBeginDate', u'payEndDate', u'listDate', u'raiseCap', u'frRaiseCap', u'allotCost', u'frAllotCost']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','equTypeCD': 'str','proPublishDate': 'str','allotCode': 'str','allotAbbr': 'str','currencyCD': 'str','recordDate': 'str','exRightsDate': 'str','payBeginDate': 'str','payEndDate': 'str','listDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquSPOpCCXEGet(secID = "", ticker = "", exchangeCD = "", spoTypeCD = "", equTypeCD = "", changeTypeCD = "", field = "", pandas = "1"):
    """
    获取历次增发预案及变动情况。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param spoTypeCD: 增发类型。例如，0101-公开增发；0102-非公开增发。对应DataAPI.SysCodeGet.codeTypeID=20019。,可以是列表,可空
    :param equTypeCD: 发行股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
    :param changeTypeCD: 方案变动类型。例如，01-未变动；02-变动。对应DataAPI.SysCodeGet.codeTypeID=20005。,可以是列表,可空
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
    requestString.append('/api/equity/getEquSPOpCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&spoTypeCD=")
    if hasattr(spoTypeCD,'__iter__') and not isinstance(spoTypeCD, str):
        if len(spoTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = spoTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(spoTypeCD))
    else:
        requestString.append(spoTypeCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
    requestString.append("&changeTypeCD=")
    if hasattr(changeTypeCD,'__iter__') and not isinstance(changeTypeCD, str):
        if len(changeTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = changeTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(changeTypeCD))
    else:
        requestString.append(changeTypeCD)
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
            api_base.handle_error(csvString, 'EquSPOpCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquSPOpCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'serialNum', u'eventNum', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'planPublishDate', u'spoTypeCD', u'equTypeCD', u'vldBeginDate', u'vldEndDate', u'issuePricePul', u'issuePricePll', u'issueSharesPul', u'issueSharesPll', u'changeTypeCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','planPublishDate': 'str','spoTypeCD': 'str','equTypeCD': 'str','vldBeginDate': 'str','vldEndDate': 'str','changeTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquSPOiCCXEGet(secID = "", ticker = "", exchangeCD = "", spoTypeCD = "", equTypeCD = "", field = "", pandas = "1"):
    """
    获取历次增发的实施信息，包括发行价、发行量、发行费用的相关信息。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param spoTypeCD: 增发类型。例如，0101-公开增发；0102-非公开增发。对应DataAPI.SysCodeGet.codeTypeID=20019。,可以是列表,可空
    :param equTypeCD: 发行股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
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
    requestString.append('/api/equity/getEquSPOiCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&spoTypeCD=")
    if hasattr(spoTypeCD,'__iter__') and not isinstance(spoTypeCD, str):
        if len(spoTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = spoTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(spoTypeCD))
    else:
        requestString.append(spoTypeCD)
    requestString.append("&equTypeCD=")
    if hasattr(equTypeCD,'__iter__') and not isinstance(equTypeCD, str):
        if len(equTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = equTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(equTypeCD))
    else:
        requestString.append(equTypeCD)
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
            api_base.handle_error(csvString, 'EquSPOiCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquSPOiCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'serialNum', u'eventNum', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'spoTypeCD', u'equTypeCD', u'issueBeginDate', u'issueEndDate', u'listPublishDate', u'recordDate', u'exRightsDate', u'listDate', u'issuePrice', u'frIssuePrice', u'currencyCD', u'issueShares', u'issueRaiseCap', u'frIssueRaiseCap', u'tradeShares', u'allotmentRatio']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','spoTypeCD': 'str','equTypeCD': 'str','issueBeginDate': 'str','issueEndDate': 'str','listPublishDate': 'str','recordDate': 'str','exRightsDate': 'str','listDate': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquRefdCCXEGet(secID = "", exchangeCD = "", ticker = "", eventProcessCD = "", isPass = "", field = "", pandas = "1"):
    """
    获取股票股权分置改革的基本信息，包含股改进程、股改实施方案以及流通股的变动情况。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、exchangeCD、ticker至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、exchangeCD、ticker至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、exchangeCD、ticker至少选择一个
    :param eventProcessCD: 事件进程。例如，1-董事会预案；2-股东大会通过；3-股东大会否决。对应DataAPI.SysCodeGet.codeTypeID=20001。,可以是列表,可空
    :param isPass: 改革方案是否通过。1-通过；0-未通过。,可以是列表,可空
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
    requestString.append('/api/equity/getEquRefdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&eventProcessCD=")
    if hasattr(eventProcessCD,'__iter__') and not isinstance(eventProcessCD, str):
        if len(eventProcessCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = eventProcessCD
            requestString.append(None)
        else:
            requestString.append(','.join(eventProcessCD))
    else:
        requestString.append(eventProcessCD)
    requestString.append("&isPass=")
    if hasattr(isPass,'__iter__') and not isinstance(isPass, str):
        if len(isPass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isPass
            requestString.append(None)
        else:
            requestString.append(','.join(isPass))
    else:
        requestString.append(isPass)
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
            api_base.handle_error(csvString, 'EquRefdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquRefdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'partyID', u'exchangeCD', u'ticker', u'secShortName', u'secShortNameEn', u'iniPublishDate', u'latPublishDate', u'isPass', u'eventProcessCD', u'adjTypeCD', u'imPublishDate', u'recordDate', u'haltDate', u'af1stTradeDate', u'cshareListDate', u'bfShares', u'bfNontradeShares', u'bfTradeShares', u'bfTradeAshares', u'afShares', u'afRestShares', u'afNonrestShares', u'afNonrestAshares']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','secShortNameEn': 'str','iniPublishDate': 'str','latPublishDate': 'str','eventProcessCD': 'str','adjTypeCD': 'str','imPublishDate': 'str','recordDate': 'str','haltDate': 'str','af1stTradeDate': 'str','cshareListDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquRefcCCXEGet(secID = "", ticker = "", exchangeCD = "", isEffected = "", capTypeCD = "", field = "", pandas = "1"):
    """
    获取股票股权分置改革支付对价方案的具体信息，包含该方案是否生效以及具体的支付比例。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param isEffected: 是否有效。1-是，0-否。,可以是列表,可空
    :param capTypeCD: 获付股东股本性质。例如，1-流通A股；2-H股；3-B股。对应DataAPI.SysCodeGet.codeTypeID=20013。,可以是列表,可空
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
    requestString.append('/api/equity/getEquRefcCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&isEffected=")
    if hasattr(isEffected,'__iter__') and not isinstance(isEffected, str):
        if len(isEffected) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isEffected
            requestString.append(None)
        else:
            requestString.append(','.join(isEffected))
    else:
        requestString.append(isEffected)
    requestString.append("&capTypeCD=")
    if hasattr(capTypeCD,'__iter__') and not isinstance(capTypeCD, str):
        if len(capTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = capTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(capTypeCD))
    else:
        requestString.append(capTypeCD)
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
            api_base.handle_error(csvString, 'EquRefcCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquRefcCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'partyID', u'exchangeCD', u'ticker', u'secShortName', u'secShortNameEn', u'iniPublishDate', u'latPublishDate', u'isEffected', u'capTypeCD', u'perShareDivRatio', u'sharePayNtsh', u'comPayShare', u'perShareTransRatio', u'ntshTransShare', u'comTransShare', u'perCashDiv', u'ntshPayCash', u'comPayCash', u'perCashDivAfTax', u'warrantRatio', u'ntshPayWarrant', u'comPayWarrant', u'comDtShares', u'comCash', u'comWarrant', u'bfNtsharesRatio', u'afNtsharesRatio', u'shareConsiRatio', u'perCashConsi', u'warrantConsiRatio', u'shareConsi', u'cashConsi', u'warrantConsi']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','secShortNameEn': 'str','iniPublishDate': 'str','latPublishDate': 'str','capTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquShareCCXEGet(secID = "", ticker = "", exchangeCD = "", changeTypeCD = "", changeDateStart = "", changeDateEnd = "", field = "", pandas = "1"):
    """
    获取上市公司最新股本结构及历次股本各部分变动情况数据。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param changeTypeCD: 股本变动原因类别。例如，01-未变动；02-变动。对应DataAPI.SysCodeGet.codeTypeID=20005。,可以是列表,可空
    :param changeDateStart: 股本发生变更日期，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有股本发生变动的记录。,可空
    :param changeDateEnd: 股本发生变更日期，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间所有股本发生变动的记录。,可空
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
    requestString.append('/api/equity/getEquShareCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&changeTypeCD=")
    if hasattr(changeTypeCD,'__iter__') and not isinstance(changeTypeCD, str):
        if len(changeTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = changeTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(changeTypeCD))
    else:
        requestString.append(changeTypeCD)
    if not isinstance(changeDateStart, str) and not isinstance(changeDateStart, unicode):
        changeDateStart = str(changeDateStart)

    requestString.append("&changeDateStart=%s"%(changeDateStart))
    if not isinstance(changeDateEnd, str) and not isinstance(changeDateEnd, unicode):
        changeDateEnd = str(changeDateEnd)

    requestString.append("&changeDateEnd=%s"%(changeDateEnd))
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
            api_base.handle_error(csvString, 'EquShareCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquShareCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'partyID', u'exchangeCD', u'ticker', u'secShortName', u'secShortNameEn', u'publishDate', u'changeDate', u'totalShares', u'aShares', u'bShares', u'hShares', u'sShares', u'nShares', u'otherShares', u'nonrestFloatShares', u'nonrestFloatA', u'floatB', u'restShares', u'restA', u'restB', u'nonListedShares', u'nonListedA', u'nonListedB', u'changeTypeCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','secShortNameEn': 'str','publishDate': 'str','changeDate': 'str','changeTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMainshFCCXEGet(secID = "", ticker = "", exchangeCD = "", shRank = "", endDateStart = "", endDateEnd = "", shareCharType = "", field = "", pandas = "1"):
    """
    获取公司十大流通股东历次变动记录，包括主要股东构成及持股数量比例、持股性质。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param shRank: 十大流通股东持股排名，如输入1，得到第一大流通股东信息。,可以是列表,可空
    :param endDateStart: 本次披露的十大流通股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
    :param endDateEnd: 本次披露的十大流通股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
    :param shareCharType: 股份性质类别。例如, 0101-流通A股；0102-流通B股。对应DataAPI.SysCodeGet.codeTypeID=20015。,可以是列表,可空
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
    requestString.append('/api/equity/getEquMainshFCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&shRank=")
    if hasattr(shRank,'__iter__') and not isinstance(shRank, str):
        if len(shRank) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = shRank
            requestString.append(None)
        else:
            requestString.append(','.join(shRank))
    else:
        requestString.append(shRank)
    if not isinstance(endDateStart, str) and not isinstance(endDateStart, unicode):
        endDateStart = str(endDateStart)

    requestString.append("&endDateStart=%s"%(endDateStart))
    if not isinstance(endDateEnd, str) and not isinstance(endDateEnd, unicode):
        endDateEnd = str(endDateEnd)

    requestString.append("&endDateEnd=%s"%(endDateEnd))
    requestString.append("&shareCharType=")
    if hasattr(shareCharType,'__iter__') and not isinstance(shareCharType, str):
        if len(shareCharType) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = shareCharType
            requestString.append(None)
        else:
            requestString.append(','.join(shareCharType))
    else:
        requestString.append(shareCharType)
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
            api_base.handle_error(csvString, 'EquMainshFCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMainshFCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'endDate', u'shNum', u'shRank', u'shName', u'holdVol', u'holdPct', u'shareCharType']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','endDate': 'str','shName': 'str','shareCharType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMainshCCXEGet(secID = "", ticker = "", exchangeCD = "", shareCharType = "", shRank = "", endDateStart = "", endDateEnd = "", field = "", pandas = "1"):
    """
    获取公司十大股东历次变动记录，包括主要股东构成及持股数量比例、持股性质。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param shareCharType: 股份性质类别。例如, 0101-流通A股；0102-流通B股。对应DataAPI.SysCodeGet.codeTypeID=20015。,可以是列表,可空
    :param shRank: 十大股东持股排名，如输入1，得到第一大股东信息。,可以是列表,可空
    :param endDateStart: 本次披露的十大股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
    :param endDateEnd: 本次披露的十大股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
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
    requestString.append('/api/equity/getEquMainshCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&shareCharType=")
    if hasattr(shareCharType,'__iter__') and not isinstance(shareCharType, str):
        if len(shareCharType) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = shareCharType
            requestString.append(None)
        else:
            requestString.append(','.join(shareCharType))
    else:
        requestString.append(shareCharType)
    requestString.append("&shRank=")
    if hasattr(shRank,'__iter__') and not isinstance(shRank, str):
        if len(shRank) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = shRank
            requestString.append(None)
        else:
            requestString.append(','.join(shRank))
    else:
        requestString.append(shRank)
    if not isinstance(endDateStart, str) and not isinstance(endDateStart, unicode):
        endDateStart = str(endDateStart)

    requestString.append("&endDateStart=%s"%(endDateStart))
    if not isinstance(endDateEnd, str) and not isinstance(endDateEnd, unicode):
        endDateEnd = str(endDateEnd)

    requestString.append("&endDateEnd=%s"%(endDateEnd))
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
            api_base.handle_error(csvString, 'EquMainshCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMainshCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'endDate', u'shNum', u'shRank', u'shName', u'holdVol', u'holdPct', u'shareCharType', u'holdRestVol', u'pleFrzCus', u'pleVol', u'frzVol', u'cusVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','endDate': 'str','shName': 'str','shareCharType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMsChanCCXEGet(secID = "", ticker = "", exchangeCD = "", relationship = "", changePctLl = "", changePctUl = "", field = "", pandas = "1"):
    """
    获取上市公司高管及其亲属买卖所在公司股份的情况，包括持股人与上市公司高管关系以及变动比例。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param relationship: 股份变动人与高管关系，如本人、父母等。,可空
    :param changePctLl: 股份变动人持股变动比例下限,可空
    :param changePctUl: 股份变动人持股变动比例上限,可空
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
    requestString.append('/api/equity/getEquMsChanCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    if not isinstance(relationship, str) and not isinstance(relationship, unicode):
        relationship = str(relationship)

    requestString.append("&relationship=%s"%(relationship))
    if not isinstance(changePctLl, str) and not isinstance(changePctLl, unicode):
        changePctLl = str(changePctLl)

    requestString.append("&changePctLl=%s"%(changePctLl))
    if not isinstance(changePctUl, str) and not isinstance(changePctUl, unicode):
        changePctUl = str(changePctUl)

    requestString.append("&changePctUl=%s"%(changePctUl))
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
            api_base.handle_error(csvString, 'EquMsChanCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMsChanCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'reportDate', u'changeDate', u'serialNum', u'managerName', u'position', u'shName', u'relationship', u'shareCharType', u'sharesChange', u'avgPrice', u'changePct', u'sharesBfChange', u'sharesAfChange', u'changeReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','reportDate': 'str','changeDate': 'str','managerName': 'str','position': 'str','shName': 'str','relationship': 'str','shareCharType': 'str','changeReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquPunishCCXEGet(secID = "", ticker = "", exchangeCD = "", punishType = "", isIrreg = "", punishReason = "", field = "", pandas = "1"):
    """
    获取公司违规受处罚事项描述说明。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param punishType: 处罚类型，如罚款，警告等。,可空
    :param isIrreg: 本次处罚事件中，本上市公司是否违规。1-是，0-否。,可以是列表,可空
    :param punishReason: 被处罚原因，如信息披露虚假，违规发行等。,可空
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
    requestString.append('/api/equity/getEquPunishCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    if not isinstance(punishType, str) and not isinstance(punishType, unicode):
        punishType = str(punishType)

    requestString.append("&punishType=%s"%(punishType))
    requestString.append("&isIrreg=")
    if hasattr(isIrreg,'__iter__') and not isinstance(isIrreg, str):
        if len(isIrreg) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isIrreg
            requestString.append(None)
        else:
            requestString.append(','.join(isIrreg))
    else:
        requestString.append(isIrreg)
    if not isinstance(punishReason, str) and not isinstance(punishReason, unicode):
        punishReason = str(punishReason)

    requestString.append("&punishReason=%s"%(punishReason))
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
            api_base.handle_error(csvString, 'EquPunishCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquPunishCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'publishDate', u'infoSource', u'punishOrg', u'punishSum', u'isIrreg', u'punishReason', u'punishType', u'comPunishSum']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','publishDate': 'str','infoSource': 'str','punishOrg': 'str','punishReason': 'str','punishType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquPunishdCCXEGet(secID = "", ticker = "", exchangeCD = "", objRelationship = "", punishReason = "", punishType = "", field = "", pandas = "1"):
    """
    获取公司违规受处罚事项明细信息说明。
    
    :param secID: 一只证券代码,格式是“数字.交易所代码”，如000001.XSHE。,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只股票代码，如000001。,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param objRelationship: 处罚对象与上市公司关联关系，可选择本公司、母公司、股东等。,可空
    :param punishReason: 被处罚原因，如信息披露虚假，违规发行等。,可空
    :param punishType: 处罚类型，如罚款，警告等。,可空
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
    requestString.append('/api/equity/getEquPunishdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    if not isinstance(objRelationship, str) and not isinstance(objRelationship, unicode):
        objRelationship = str(objRelationship)

    requestString.append("&objRelationship=%s"%(objRelationship))
    if not isinstance(punishReason, str) and not isinstance(punishReason, unicode):
        punishReason = str(punishReason)

    requestString.append("&punishReason=%s"%(punishReason))
    if not isinstance(punishType, str) and not isinstance(punishType, unicode):
        punishType = str(punishType)

    requestString.append("&punishType=%s"%(punishType))
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
            api_base.handle_error(csvString, 'EquPunishdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquPunishdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'seq', u'punishSeq', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secShortNameEn', u'publishDate', u'punishObj', u'objRelationship', u'punishReason', u'punishType', u'punishSum']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','publishDate': 'str','punishObj': 'str','objRelationship': 'str','punishReason': 'str','punishType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBlockdeCCXEGet(secID = "", ticker = "", exchangeCD = "", assetClass = "", tradeDate = "", field = "", pandas = "1"):
    """
    获取上海、深圳交易所公布的每个交易日股票大宗交易具体信息，包括证券信息、成交价、成交量以及买卖双方营业部数据。本交易日可获取前一交易日的数据。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param assetClass: 证券类型。例如，E-股票,B-债券,F-基金。对应DataAPI.SysCodeGet.codeTypeID=10001。,可以是列表,可空
    :param tradeDate: 交易日期，默认为前1天，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/equity/getMktBlockdeCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&assetClass=")
    if hasattr(assetClass,'__iter__') and not isinstance(assetClass, str):
        if len(assetClass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = assetClass
            requestString.append(None)
        else:
            requestString.append(','.join(assetClass))
    else:
        requestString.append(assetClass)
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
            api_base.handle_error(csvString, 'MktBlockdeCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBlockdeCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'tradeDate', u'secID', u'ticker', u'assetClass', u'exchangeCD', u'secShortName', u'secShortNameEn', u'tradePrice', u'tradeVal', u'tradeVol', u'buyerBd', u'sellerBd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','secID': 'str','ticker': 'str','assetClass': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','buyerBd': 'str','sellerBd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBlockbszCCXEGet(secID = "", ticker = "", exchangeCD = "", assetClass = "", tradeDate = "", field = "", pandas = "1"):
    """
    获取深圳交易所公布的每个交易日债券大宗交易具体信息，包括证券信息、成交价、成交量以及买卖双方营业部数据。本交易日可获取前一交易日的数据。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param assetClass: 证券类型。例如，E-股票,B-债券,F-基金。对应DataAPI.SysCodeGet.codeTypeID=10001。,可以是列表,可空
    :param tradeDate: 交易日期，默认为前1天，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/equity/getMktBlockbszCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&assetClass=")
    if hasattr(assetClass,'__iter__') and not isinstance(assetClass, str):
        if len(assetClass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = assetClass
            requestString.append(None)
        else:
            requestString.append(','.join(assetClass))
    else:
        requestString.append(assetClass)
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
            api_base.handle_error(csvString, 'MktBlockbszCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBlockbszCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'tradeDate', u'secID', u'ticker', u'assetClass', u'exchangeCD', u'secShortName', u'secShortNameEn', u'tradePrice', u'tradeVal', u'tradeVol', u'buyerBd', u'sellerBd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','secID': 'str','ticker': 'str','assetClass': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','buyerBd': 'str','sellerBd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBlockbshCCXEGet(secID = "", ticker = "", exchangeCD = "", assetClass = "", tradeDate = "", field = "", pandas = "1"):
    """
    获取上海交易所公布的每个交易日债券大宗交易具体信息，包括证券信息、成交价以及成交量。本交易日可获取前一交易日的数据。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param assetClass: 证券类型。例如，E-股票,B-债券,F-基金。对应DataAPI.SysCodeGet.codeTypeID=10001。,可以是列表,可空
    :param tradeDate: 交易日期，默认为前1天，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/equity/getMktBlockbshCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&assetClass=")
    if hasattr(assetClass,'__iter__') and not isinstance(assetClass, str):
        if len(assetClass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = assetClass
            requestString.append(None)
        else:
            requestString.append(','.join(assetClass))
    else:
        requestString.append(assetClass)
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
            api_base.handle_error(csvString, 'MktBlockbshCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBlockbshCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'tradeDate', u'secID', u'ticker', u'assetClass', u'exchangeCD', u'secShortName', u'secShortNameEn', u'tradePrice', u'tradeVal', u'tradeVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','secID': 'str','ticker': 'str','assetClass': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBlockdfCCXEGet(secID = "", ticker = "", exchangeCD = "", assetClass = "", tradeDate = "", field = "", pandas = "1"):
    """
    获取上海、深圳交易所公布的每个交易日基金大宗交易具体信息，包括证券信息、成交价、成交量以及买卖双方营业部数据。本交易日可获取前一交易日的数据。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param assetClass: 证券类型。例如，E-股票,B-债券,F-基金。对应DataAPI.SysCodeGet.codeTypeID=10001。,可以是列表,可空
    :param tradeDate: 交易日期，默认为前1天，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/equity/getMktBlockdfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&assetClass=")
    if hasattr(assetClass,'__iter__') and not isinstance(assetClass, str):
        if len(assetClass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = assetClass
            requestString.append(None)
        else:
            requestString.append(','.join(assetClass))
    else:
        requestString.append(assetClass)
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
            api_base.handle_error(csvString, 'MktBlockdfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBlockdfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'tradeDate', u'secID', u'ticker', u'assetClass', u'exchangeCD', u'secShortName', u'secShortNameEn', u'tradePrice', u'tradeVal', u'tradeVol', u'buyerBd', u'sellerBd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','secID': 'str','ticker': 'str','assetClass': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','buyerBd': 'str','sellerBd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBondRepodCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取交易所债券质押式回购行情，包括证券内部编码,回购代码,回购简称,回购英文简称,交易市场,交易日期,前收盘利率,开盘利率,最高利率,最低利率,收盘利率,成交额,成交量,成交笔数等，历史数据追溯至1994年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'REPO',输入'131801'，可获取到secID'131801.XSHE'后，在此输入'131801.XSHE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktBondRepodCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktBondRepodCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBondRepodCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preCloseRate', u'openRate', u'highestRate', u'lowestRate', u'closeRate', u'turnoverValue', u'turnovervVol', u'dealAmount']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBondbYdIdxdCCXEGet(startDate = "", endDate = "", publishDate = "", field = "", pandas = "1"):
    """
    获取长三角票据贴现指数反映长三角地区票据贴现市场平均价格水平和总体变化趋势，包含开始日期，结束日期，发布日期，指数点位，涨跌幅等，历史追溯至2013年1月30日，每日更新。
    
    :param startDate: 开始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
    :param publishDate: 发布日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktBondbYdIdxdCCXE.csv?ispandas=1&') 
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(publishDate, str) and not isinstance(publishDate, unicode):
        publishDate = str(publishDate)

    requestString.append("&publishDate=%s"%(publishDate))
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
            api_base.handle_error(csvString, 'MktBondbYdIdxdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBondbYdIdxdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'startDate', u'endDate', u'publishDate', u'indexPoint', u'chg']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'startDate': 'str','endDate': 'str','publishDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBonddCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取交易所债券日行情，包含证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,昨收价,开盘价,最高价,最低价,收盘价,成交金额,成交数量,成交笔数,应计利息,到期收益率等，历史追溯至1993年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'b',输入'010107'，可获取到secID'010107.XSHG'后，在此输入'010107.XSHG'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktBonddCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktBonddCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBonddCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverValue', u'turnoverVol', u'dealAmount', u'accrInterest', u'YTM']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIBBondOrepodCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取银行间买断式回购交易行情，包含证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,前收盘利率,前加权平均利率,开盘利率,最高利率,最低利率,收盘利率,加权平均利率,成交额,成交量等，历史追溯至2004年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'repo',输入'OR001'，可获取到secID'OR001.XIBE'后，在此输入'OR001.XIBE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktIBBondOrepodCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktIBBondOrepodCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIBBondOrepodCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preCloseRate', u'preCloseWeiRate', u'openRate', u'highestRate', u'lowestRate', u'closeRate', u'wRate', u'turnoverValue', u'turnoverVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIBBondOreposdCCXEGet(secID, repoIssueType = "", startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取银行间买断式回购结算行情，包括证券内部编码,证券代码,证券简称,证券英文简称,交易市场,回购发行人类型,交易日期,前期平均利率,期初利率,最高利率,最低利率,期末利率,本期平均利率,涨跌幅,3个月滑动平均利率,6个月滑动平均利率,12个月滑动平均利率,内含期限,本金额,面值额,资金额,结算笔数,平均每笔交割量,结算天数等，历史数据追溯至2004年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'repo',输入'OR001'，可获取到secID'OR001.XIBE'后，在此输入'OR001.XIBE'
    :param repoIssueType: 回购发行人类型，具体为0为全部债权发行人，1为财政部，2为国家开发银行，3为中国进出口银行，4为中国人民银行总行，5为中信信托，6为其他，7为中国农业发展银行，8为中国铁路总公司,可空
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktIBBondOreposdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(repoIssueType, str) and not isinstance(repoIssueType, unicode):
        repoIssueType = str(repoIssueType)

    requestString.append("&repoIssueType=%s"%(repoIssueType))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktIBBondOreposdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIBBondOreposdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'repoIssueType', u'tradeDate', u'preAVGRate', u'preRate', u'highestRate', u'lowestRate', u'finRate', u'avgRate', u'chgPct', u'slideAvgRate3m', u'slideAvgRate6m', u'slideAvgRate12m', u'incLimited', u'principal', u'par', u'capital', u'dealAmount', u'avgDealVol', u'settlDay']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','repoIssueType': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIBBondRepodCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取银行间质押式回购交易行情，包括证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,前收盘利率,前加权平均利率,开盘利率,最高利率,最低利率,收盘利率,加权平均利率,成交额,成交量等，历史追溯至1997年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'repo',输入'R001'，可获取到secID'R001.XIBE'后，在此输入'R001.XIBE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktIBBondRepodCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktIBBondRepodCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIBBondRepodCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preCloseRate', u'preCloseWeiRate', u'openRate', u'highestRate', u'lowestRate', u'closeRate', u'wRate', u'turnoverValue', u'turnoverVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIBBondReposdCCXEGet(secID, repoIssueType = "", startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取银行间质押式回购结算行情，包括证券内部编码,证券代码,证券简称,证券英文简称,交易市场,回购发行人类型,交易日期,前期平均利率,期初利率,最高利率,最低利率,期末利率,本期平均利率,涨跌幅,3个月滑动平均利率,6个月滑动平均利率,12个月滑动平均利率,内含期限,本金额,面值额,资金额,结算笔数,平均每笔交割量,结算天数等，历史数据追溯至1999年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'repo',输入'R001'，可获取到secID'R001.XIBE'后，在此输入'R001.XIBE'
    :param repoIssueType: 回购发行人类型，具体为0为全部债权发行人，1为财政部，2为国家开发银行，3为中国进出口银行，4为中国人民银行总行，5为中信信托，6为其他，7为中国农业发展银行，8为中国铁路总公司,可空
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktIBBondReposdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(repoIssueType, str) and not isinstance(repoIssueType, unicode):
        repoIssueType = str(repoIssueType)

    requestString.append("&repoIssueType=%s"%(repoIssueType))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktIBBondReposdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIBBondReposdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'repoIssueType', u'tradeDate', u'preAVGRate', u'preRate', u'highestRate', u'lowestRate', u'finRate', u'avgRate', u'chgPct', u'slideAvgRate3m', u'slideAvgRate6m', u'slideAvgRate12m', u'incLimited', u'principal', u'par', u'capital', u'dealAmount', u'avgDealVol', u'settlDay']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','repoIssueType': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIBBonddCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取银行间现券交易行情，包括证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,前收盘净价,前加权平均净价,开盘净价,最高净价,最低净价,收盘净价,加权平均净价,净价涨跌幅,前收盘全价,前加权平均全价,开盘全价,最高全价,最低全价,收盘全价,加权平均全价,全价涨跌幅,券面总额,成交量,前收盘收益率,前加权平均收益率,开盘收益率,最高收益率,最低收益率,收盘收益率,加权平均收益率等，历史数据追溯至1997年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'b',输入'140444'，可获取到secID'140444.XIBE'后，在此输入'140444.XIBE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktIBBonddCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktIBBonddCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIBBonddCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preCloseNetPrice', u'preWAvgNetPrice', u'openNetPrice', u'highestNetPrice', u'lowestNetPrice', u'closeNetPrice', u'wAvgNetPrice', u'netPriceChgPct', u'preClosePrice', u'preWAvgPrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'WAvgPrice', u'chgPct', u'ternoverValue', u'turnoverVol', u'yieldPreClosePrice', u'yieldPreWAvgPrice', u'yieldOpenPrice', u'yieleHighestPrice', u'YieldLowestPrice', u'yieldClosePrice', u'yieldWAvgPrice']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktIBBondsdCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取存储中央国债登记结算公司发布的，每个结算日的结算行情，包括证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,期末应计利息,待偿期,平均年收益率,前期平均净价,期初结算净价,最高结算净价,最低结算净价,期末结算净价,本期平均净价,平均净价涨跌幅,前期平均全价,期初结算全价,最高结算全价,最低结算全价,期末结算全价,本期平均全价1999年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'b',输入'100020'，可获取到secID'100020.XIBE'后，在此输入'100020.XIBE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/bond/getMktIBBondsdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktIBBondsdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktIBBondsdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'finAccrInterest', u'accrLimited', u'avgYieldYoy', u'preAvgNetPrice', u'preSettleNetPrice', u'highestSettleNetPrice', u'lowestSettleNetPrice', u'finSettleNetPrice', u'avgNetPrice', u'netPriceChgPct', u'preAvgPrice', u'preSettlePrice', u'highestSettlePrice', u'lowestSettlePrice', u'finSettlePrice', u'avgPrice', u'chgPct', u'principal', u'par', u'capital', u'dealAmount', u'avgDealVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktObondBilRediCCXEGet(startDate = "", endDate = "", tradeType = "", field = "", pandas = "1"):
    """
    获取每个交易日票据市场转贴现回购报价行情，分纸票回购报价和电票回购报价两种，包含交易日期,票据形态,交易方向,票据类型,交易序号,金额,利率,剩余,承兑行类型等，历史追溯至2013年1月30日，每日更新。
    
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
    :param tradeType: 交易序号，自增长,可空
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
    requestString.append('/api/bond/getMktObondBilRediCCXE.csv?ispandas=1&') 
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(tradeType, str) and not isinstance(tradeType, unicode):
        tradeType = str(tradeType)

    requestString.append("&tradeType=%s"%(tradeType))
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
            api_base.handle_error(csvString, 'MktObondBilRediCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktObondBilRediCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'tradeDate', u'billForm', u'tradeType', u'billType', u'tradeNum', u'billAmount', u'intRate', u'maturity', u'bankType']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','billForm': 'str','tradeType': 'str','billType': 'str','maturity': 'str','bankType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBondBillRediCCXEGet(startDate = "", endDate = "", tradeType = "", field = "", pandas = "1"):
    """
    获取每个交易日票据市场转贴现买断报价行情，分纸票买断报价和电票买断报价两种，包含交易日期,票据形态,交易方向,票据类型,交易序号,金额,利率,期限等，历史追溯至2013年1月30日，每日更新。
    
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
    :param tradeType: 交易序号，自增长,可空
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
    requestString.append('/api/bond/getMktBondBillRediCCXE.csv?ispandas=1&') 
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(tradeType, str) and not isinstance(tradeType, unicode):
        tradeType = str(tradeType)

    requestString.append("&tradeType=%s"%(tradeType))
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
            api_base.handle_error(csvString, 'MktBondBillRediCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBondBillRediCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'tradeDate', u'billForm', u'tradeType', u'billType', u'tradeNum', u'billAmount', u'intRate', u'maturity', u'bankType']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','billForm': 'str','tradeType': 'str','billType': 'str','maturity': 'str','bankType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondAiCCXEGet(secID = "", ticker = "", dataStartDate = "", dataEndDate = "", field = "", pandas = "1"):
    """
    1.计算债券从起息日到当前日期，每个自然日的债券的代偿期、应计利息。若债券已到期，则计算只计算债券存续期内的；2.历史数据追溯至1990年，每日更新。
    
    :param secID: 证券ID,secID、ticker至少选择一个
    :param ticker: 交易代码,secID、ticker至少选择一个
    :param dataStartDate: 查询起始日期，输入格式“YYYYMMDD”,可空
    :param dataEndDate: 查询截止日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/bond/getBondAiCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    if not isinstance(dataStartDate, str) and not isinstance(dataStartDate, unicode):
        dataStartDate = str(dataStartDate)

    requestString.append("&dataStartDate=%s"%(dataStartDate))
    if not isinstance(dataEndDate, str) and not isinstance(dataEndDate, unicode):
        dataEndDate = str(dataEndDate)

    requestString.append("&dataEndDate=%s"%(dataEndDate))
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
            api_base.handle_error(csvString, 'BondAiCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondAiCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'dataDate', u'perAccrDate', u'days', u'ai', u'aiPar', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','dataDate': 'str','perAccrDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondCfCCXEGet(secID = "", ticker = "", cashTypeCD = "", paymentDateBegin = "", paymentDateEnd = "", field = "", pandas = "1"):
    """
    1、存储债券理论计算的各期本金、利息支付，若含权，单独存储其行权期的本金、利息支付。当实际行权时，根据公告进行更新还本付息；2、分期还本债券的利息，是根据还本后实际百元剩余面额占面值的比例来计算的；3、该表未含分期还本的资产支持证券和私募债券的还本记录。原因是公告对分期还本资产支持型证券和私募债券的还本期次未明确披露；4、对于可转债，由于回售、赎回与公司股价波动有关，行权期不确定，因此理论计算时按照到期还本来处理；5、历史数据追溯至1981年，每日更新。
    
    :param secID: 证券ID,secID、ticker至少选择一个
    :param ticker: 交易代码,secID、ticker至少选择一个
    :param cashTypeCD: 现金流类型。INTPAY-付息；CAPINTPAY-还本付息。对应DataAPI.SysCodeGet.codeTypeID=30014。,可以是列表,可空
    :param paymentDateBegin: 现金流起始日，输入格式“YYYYMMDD”,可空
    :param paymentDateEnd: 现金流截止日，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/bond/getBondCfCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    requestString.append("&cashTypeCD=")
    if hasattr(cashTypeCD,'__iter__') and not isinstance(cashTypeCD, str):
        if len(cashTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = cashTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(cashTypeCD))
    else:
        requestString.append(cashTypeCD)
    if not isinstance(paymentDateBegin, str) and not isinstance(paymentDateBegin, unicode):
        paymentDateBegin = str(paymentDateBegin)

    requestString.append("&paymentDateBegin=%s"%(paymentDateBegin))
    if not isinstance(paymentDateEnd, str) and not isinstance(paymentDateEnd, unicode):
        paymentDateEnd = str(paymentDateEnd)

    requestString.append("&paymentDateEnd=%s"%(paymentDateEnd))
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
            api_base.handle_error(csvString, 'BondCfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondCfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'cashTypeCD', u'intePeriod', u'perAccrDate', u'PERAccrEndDate', u'isOptPeriod', u'interest', u'payment', u'payAmut', u'exerInterest', u'exerPayment', u'exerPayAmut', u'paymentDate', u'payEndDate', u'recordDate', u'exDivDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','cashTypeCD': 'str','perAccrDate': 'str','PERAccrEndDate': 'str','paymentDate': 'str','payEndDate': 'str','recordDate': 'str','exDivDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondConvHolderCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1、可转换债券持有人的持券情况；2、历史数据追溯至2000年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondConvHolderCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondConvHolderCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondConvHolderCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'secShortName', u'publishDate', u'endDate', u'holderSN', u'holdRank', u'holderName', u'holdAmount', u'holdVol', u'holdRatio', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDate': 'str','holderName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondConvPlaceCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1、存储可转债发行配售时，具体各股东的配售数量、金额等信息；2、历史数据追溯至2002年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondConvPlaceCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondConvPlaceCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondConvPlaceCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'secShortName', u'publishDate', u'SN', u'partyFullName', u'placeAmount', u'placeVol', u'retuAmount', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','partyFullName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondConvCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1、存储可转债的转股信息以及由转股、赎回、回售引起的规模变动情况；2、历史数据追溯至2000年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondConvCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondConvCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondConvCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'secShortName', u'publishDate', u'endDate', u'convEventTypeCD', u'issueAmount', u'initialConvPrice', u'thisConvPrice', u'thisConvAmount', u'thisConvVol', u'thisConvRatio', u'cumuConvAmut', u'cumuConvVol', u'cumuAveConvPrice', u'cumuConvRatio', u'convRemainAmut', u'convTotalShare', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDate': 'str','convEventTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondConvPriceChangeCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1、存储可转债发行时约定的初始转股价格，以及由公司分红、送转股等引起的历次转股价格变动的情况；2、历史数据追溯至1998年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondConvPriceChangeCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondConvPriceChangeCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondConvPriceChangeCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'secShortName', u'publishDate', u'convDate', u'convPrice', u'chanReason', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','convDate': 'str','chanReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondCouponCCXEGet(secID = "", ticker = "", perValueDate = "", perValueEndDate = "", field = "", pandas = "1"):
    """
    1、记录固定利率债券、浮动利率债券每个利息支付期间的利率情况，包括分段计息的具体利率；2、历史数据追溯至1987年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param perValueDate: 当期起息日,可空
    :param perValueEndDate: 当期结息日，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/bond/getBondCouponCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(perValueDate, str) and not isinstance(perValueDate, unicode):
        perValueDate = str(perValueDate)

    requestString.append("&perValueDate=%s"%(perValueDate))
    if not isinstance(perValueEndDate, str) and not isinstance(perValueEndDate, unicode):
        perValueEndDate = str(perValueEndDate)

    requestString.append("&perValueEndDate=%s"%(perValueEndDate))
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
            api_base.handle_error(csvString, 'BondCouponCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondCouponCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'couponTypeCD', u'cpnFreqCD', u'interestPeriod', u'perValueDate', u'perValueEndDate', u'refRatePer', u'frnMargin', u'frnRefRateCD', u'refRateDate', u'stepMargin', u'perAccrDate', u'perAccrEndDate', u'coupon', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','couponTypeCD': 'str','cpnFreqCD': 'str','perValueDate': 'str','perValueEndDate': 'str','frnRefRateCD': 'str','refRateDate': 'str','perAccrDate': 'str','perAccrEndDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondGuarCCXEGet(secID = "", ticker = "", guarModeCD = "", guarTypeCD = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    1、收录债券担保机构的名称、性质、担保方式、担保期限等；2、历史数据追溯至1996年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param guarModeCD: 担保人类型。例如，0101-保证担保；0102-抵押担保等。对应DataAPI.SysCodeGet.codeTypeID=30012。,可以是列表,可空
    :param guarTypeCD: 担保方式。例如，1-担保；3-反担保等。对应DataAPI.SysCodeGet.codeTypeID=30011。,可以是列表,可空
    :param beginDate: 查询起始日期，输入格式“YYYYMMDD”,可空
    :param endDate: 查询截止日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/bond/getBondGuarCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&guarModeCD=")
    if hasattr(guarModeCD,'__iter__') and not isinstance(guarModeCD, str):
        if len(guarModeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = guarModeCD
            requestString.append(None)
        else:
            requestString.append(','.join(guarModeCD))
    else:
        requestString.append(guarModeCD)
    requestString.append("&guarTypeCD=")
    if hasattr(guarTypeCD,'__iter__') and not isinstance(guarTypeCD, str):
        if len(guarTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = guarTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(guarTypeCD))
    else:
        requestString.append(guarTypeCD)
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'BondGuarCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondGuarCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'publishDate', u'guarModeCD', u'guar', u'guarRange', u'guarTypeCD', u'beginDate', u'endDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','publishDate': 'str','guarModeCD': 'str','guar': 'str','guarRange': 'str','guarTypeCD': 'str','beginDate': 'str','endDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondInfo1CCXEGet(secID = "", ticker = "", typeID = "", field = "", pandas = "1"):
    """
    1.存储除了私募债、可转债、国债、央票、ABS、MBS、SAMP、ABN之外的所有债券的基本信息，含企业债、公司债、短融、中期票据、金融债、国际机构债券、集合票据、中小企业集合债券等等债券类型；2.历史数据追溯至1996年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param typeID: 债券类型。例如，0202010101-国债；0202010201-央行票据。对应DataAPI.SysCodeGet.codeTypeID=30018。,可空
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
    requestString.append('/api/bond/getBondInfo1CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(typeID, str) and not isinstance(typeID, unicode):
        typeID = str(typeID)

    requestString.append("&typeID=%s"%(typeID))
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
            api_base.handle_error(csvString, 'BondInfo1CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondInfo1CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'issuer', u'par', u'currencyCD', u'typeID', u'couponTypeCD', u'cpnFreqCD', u'paymentCD', u'coupon', u'minCoupon', u'frnRefRateCD', u'frnCurrBmkRate', u'frnMargin', u'firstAccrDate', u'theoEndDate', u'actuEndDate', u'firstRedempDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','issuer': 'str','currencyCD': 'str','typeID': 'str','couponTypeCD': 'str','cpnFreqCD': 'str','paymentCD': 'str','frnRefRateCD': 'str','firstAccrDate': 'str','theoEndDate': 'str','actuEndDate': 'str','firstRedempDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondInfo2CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.存储可转债、分离交易可转债的基本信息，如利率、转股的信息；2.历史数据追溯至1991年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondInfo2CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondInfo2CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondInfo2CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'partyID', u'issuer', u'hybridCD', u'par', u'currencyCD', u'couponTypeCD', u'cpnFreqCD', u'paymentCD', u'coupon', u'firstAccrDate', u'theoEndDate', u'actuEndDate', u'firstRedempDate', u'convStkShortName', u'convCode', u'convStartDate', u'convEndDate', u'stopConvDate', u'callProt', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','issuer': 'str','hybridCD': 'str','currencyCD': 'str','couponTypeCD': 'str','cpnFreqCD': 'str','paymentCD': 'str','firstAccrDate': 'str','theoEndDate': 'str','actuEndDate': 'str','firstRedempDate': 'str','convStkShortName': 'str','convCode': 'str','convStartDate': 'str','convEndDate': 'str','stopConvDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondInfo3CCXEGet(secID = "", ticker = "", typeID = "", field = "", pandas = "1"):
    """
    1.存储MBS、ABS、ABN、SAMP的基本信息；2.历史数据追溯至2005年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param typeID: 债券分类。例如，0202010101-国债；0202010201-央行票据。对应DataAPI.SysCodeGet.codeTypeID=30018。,可空
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
    requestString.append('/api/bond/getBondInfo3CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(typeID, str) and not isinstance(typeID, unicode):
        typeID = str(typeID)

    requestString.append("&typeID=%s"%(typeID))
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
            api_base.handle_error(csvString, 'BondInfo3CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondInfo3CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'typeID', u'partyID', u'issuer', u'basicAsset', u'par', u'currencyCD', u'couponTypeCD', u'frnRefRateCD', u'frnCurrBmkRate', u'frnMargin', u'cpnFreqCD', u'paymentCD', u'coupon', u'firstAccrDate', u'theoEndDate', u'actuEndDate', u'firstRedempDate', u'guarName', u'isCall', u'isPut', u'absLevelCD', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','typeID': 'str','issuer': 'str','basicAsset': 'str','currencyCD': 'str','couponTypeCD': 'str','frnRefRateCD': 'str','cpnFreqCD': 'str','paymentCD': 'str','firstAccrDate': 'str','theoEndDate': 'str','actuEndDate': 'str','firstRedempDate': 'str','guarName': 'str','absLevelCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondInfo4CCXEGet(secID = "", ticker = "", typeID = "", field = "", pandas = "1"):
    """
    1.存储私募债券的发行上市相关信息；2.历史数据追溯至2012年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param typeID: 债券分类。例如，0202010101-国债；0202010201-央行票据。对应DataAPI.SysCodeGet.codeTypeID=30018。,可空
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
    requestString.append('/api/bond/getBondInfo4CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(typeID, str) and not isinstance(typeID, unicode):
        typeID = str(typeID)

    requestString.append("&typeID=%s"%(typeID))
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
            api_base.handle_error(csvString, 'BondInfo4CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondInfo4CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'typeID', u'issuer', u'par', u'currencyCD', u'couponTypeCD', u'cpnFreqCD', u'paymentCD', u'coupon', u'firstAccrDate', u'theoEndDate', u'actuEndDate', u'firstRedempDate', u'isGuar', u'isCall', u'isPut', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','typeID': 'str','issuer': 'str','currencyCD': 'str','couponTypeCD': 'str','cpnFreqCD': 'str','paymentCD': 'str','firstAccrDate': 'str','theoEndDate': 'str','actuEndDate': 'str','firstRedempDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondInfo5CCXEGet(secID = "", ticker = "", typeID = "", field = "", pandas = "1"):
    """
    1.存储国债与央行票据的基本信息；2.历史数据追溯至1987年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker、typeID至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker、typeID至少选择一个
    :param typeID: 债券分类ID。例如，0202010101-国债；0202010201-央行票据。对应DataAPI.SysCodeGet.codeTypeID=30018。,secID、ticker、typeID至少选择一个
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
    requestString.append('/api/bond/getBondInfo5CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(typeID, str) and not isinstance(typeID, unicode):
        typeID = str(typeID)

    requestString.append("&typeID=%s"%(typeID))
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
            api_base.handle_error(csvString, 'BondInfo5CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondInfo5CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'typeID', u'partyID', u'issuer', u'par', u'currencyCD', u'couponTypeCD', u'frnRefRateCD', u'frnCurrBmkRate', u'frnMargin', u'cpnFreqCD', u'paymentCD', u'coupon', u'firstAccrDate', u'theoEndDate', u'actuEndDate', u'firstRedempDate', u'isRepo', u'isCall', u'isPut', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','typeID': 'str','issuer': 'str','currencyCD': 'str','couponTypeCD': 'str','frnRefRateCD': 'str','cpnFreqCD': 'str','paymentCD': 'str','firstAccrDate': 'str','theoEndDate': 'str','actuEndDate': 'str','firstRedempDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondIssue1CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.记录普通债券在不同市场上的发行和上市信息；2.同一债券若在多个市场上市，将对应多条记录；3.历史数据追溯至1996年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondIssue1CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondIssue1CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondIssue1CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'currencyCD', u'crossExchange', u'issuePrice', u'issueDate', u'issueEndDate', u'partyID', u'issuer', u'auctionDate', u'recordDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','currencyCD': 'str','issueDate': 'str','issueEndDate': 'str','issuer': 'str','auctionDate': 'str','recordDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondIssue2CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.记录可转债券在不同市场上的发行和上市信息；2.同一债券若在多个市场上市，将对应多条记录；3.历史数据追溯至1991年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondIssue2CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondIssue2CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondIssue2CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'currencyCD', u'issuePrice', u'issueDate', u'issueEndDate', u'partyID', u'issuer', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','currencyCD': 'str','issueDate': 'str','issueEndDate': 'str','issuer': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondIssue3CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.记录资产支持证券在不同市场上的发行和上市信息；2.同一债券若在多个市场上市，将对应多条记录；3.历史数据追溯至2005年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondIssue3CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondIssue3CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondIssue3CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'currencyCD', u'crossExchange', u'issuePrice', u'issueDate', u'issueEndDate', u'partyID', u'issuer', u'auctionDate', u'recordDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','currencyCD': 'str','issueDate': 'str','issueEndDate': 'str','issuer': 'str','auctionDate': 'str','recordDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondIssue5CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.记录利率债在不同市场上的发行和上市信息；2.同一债券若在多个市场上市，将对应多条记录；3.历史数据追溯至1987年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondIssue5CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondIssue5CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondIssue5CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'currencyCD', u'crossExchange', u'issuePrice', u'issueDate', u'issueEndDate', u'partyID', u'issuer', u'auctionDate', u'recordDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','currencyCD': 'str','issueDate': 'str','issueEndDate': 'str','issuer': 'str','auctionDate': 'str','recordDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondIssueTermCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.包含可转换债券在招募说明书中列示的各类发行条款；2.历史数据追溯至1992年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondIssueTermCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondIssueTermCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondIssueTermCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'secShortName', u'publishDate', u'couponCompTerm', u'initialCpTerm', u'cpAdjTerm', u'cpRevTerm', u'forceConvTerm', u'dealRemainTerm', u'convYearDivTerm', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','couponCompTerm': 'str','initialCpTerm': 'str','cpAdjTerm': 'str','cpRevTerm': 'str','forceConvTerm': 'str','dealRemainTerm': 'str','convYearDivTerm': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondOption1CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.收录含有调换选择权、转换选择权以及合并权的债券的相关信息；2.历史数据追溯至2002年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondOption1CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondOption1CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondOption1CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'optionCD', u'optionRemark', u'ifExer', u'targetBondCode', u'targetBondShortName', u'changeRatio', u'rootOut', u'rootBalance', u'targetIn', u'targetInSum', u'exerDate', u'publishDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','optionCD': 'str','optionRemark': 'str','targetBondCode': 'str','targetBondShortName': 'str','exerDate': 'str','publishDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondOption2CCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.存储含权债券除可调换选择权、可转换选择权和合并权利之外的所有权利信息，且分为发行时权利约定及行权提示与结果；2.历史数据追溯至1992年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondOption2CCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondOption2CCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondOption2CCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'optionCD', u'isExer', u'publishDate', u'exerPrice', u'exerPriceTaxed', u'exerDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','optionCD': 'str','publishDate': 'str','exerDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondRatingCCXEGet(secID = "", ticker = "", rating = "", ratTypeCD = "", field = "", pandas = "1"):
    """
    1.债券信用分析主要介绍债券相关的信用评级信息；2.历史数据追溯至1996年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param rating: 级别,可以是列表,可空
    :param ratTypeCD: 评级类型。L-长期，S-短期。对应DataAPI.SysCodeGet.codeTypeID=30009。,可以是列表,可空
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
    requestString.append('/api/bond/getBondRatingCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&rating=")
    if hasattr(rating,'__iter__') and not isinstance(rating, str):
        if len(rating) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = rating
            requestString.append(None)
        else:
            requestString.append(','.join(rating))
    else:
        requestString.append(rating)
    requestString.append("&ratTypeCD=")
    if hasattr(ratTypeCD,'__iter__') and not isinstance(ratTypeCD, str):
        if len(ratTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ratTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(ratTypeCD))
    else:
        requestString.append(ratTypeCD)
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
            api_base.handle_error(csvString, 'BondRatingCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondRatingCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'ratingDate', u'rating', u'ratComID', u'ratCom', u'ratTypeCD', u'publishDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','ratingDate': 'str','rating': 'str','ratCom': 'str','ratTypeCD': 'str','publishDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondRepuConvRateCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.债券回购的折算比率等信息；2.历史数据追溯至2000年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondRepuConvRateCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondRepuConvRateCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondRepuConvRateCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'bondShortName', u'publishDate', u'stbConvRate', u'validStartDate', u'validEndDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','bondShortName': 'str','publishDate': 'str','validStartDate': 'str','validEndDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondRepuInfoCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.该表涵盖回购基本信息，包括回购期限、回购方式等信息；2.历史数据追溯至1995年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 回购代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondRepuInfoCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondRepuInfoCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondRepuInfoCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repoShortName', u'repoFullName', u'bondTicker', u'bondShortName', u'typeID', u'exchangeCD', u'marginRate', u'repoMatu', u'repoTypeCD', u'listDate', u'listStatus', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repoShortName': 'str','repoFullName': 'str','bondTicker': 'str','bondShortName': 'str','typeID': 'str','exchangeCD': 'str','repoTypeCD': 'str','listDate': 'str','listStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondSizeChangeCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    1.记录所有债券的规模构成及规模变动情况；2.历史数据追溯至1981年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/bond/getBondSizeChangeCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'BondSizeChangeCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondSizeChangeCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'bondShortName', u'publishDate', u'endDate', u'totalSize', u'exIBSize', u'exchangeSize', u'xshgSize', u'xsheSize', u'xibeSize', u'otcSize', u'otherSize', u'sizeChange', u'changeTypeCD', u'changeReason', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','bondShortName': 'str','publishDate': 'str','endDate': 'str','changeTypeCD': 'str','changeReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktFunddCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取交易所基金日行情，包含证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,昨收盘价,开盘价,最高价,最低价,收盘价,成交数量,成交金额等，历史追溯至1998年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'F',输入'184701'，可获取到secID'184701.XSHE'后，在此输入'184701.XSHE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/fund/getMktFunddCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktFunddCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktFunddCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol', u'turnoverValue']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundNavCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取某只基金的历史净值数据，包含单位净值、累计净值、复权单位净值、单位净值日增长率、复权单位净值日增长率等信息。收录了1998年以来的历史数据，数据更新频率为日。不输入日期则默认获取近一年以来的历史数据。注:增长率以小数形式(非百分数)提供。
    
    :param secID: 输入单个证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,secID、ticker至少选择一个
    :param ticker: 输入单个基金代码，如"000001"。,secID、ticker至少选择一个
    :param beginDate: 起始日期，输入格式为“YYYYMMDD”，不输则默认为一年前当日。,可空
    :param endDate: 截止日期,输入格式为“YYYYMMDD”，不输则默认为当日。,可空
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
    requestString.append('/api/fund/getFundNavCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
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
            api_base.handle_error(csvString, 'FundNavCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundNavCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'endDate', u'nav', u'adjustNav', u'accumNav', u'navDailyGR', u'AdjNavDailyGR']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取基金的基本档案信息，包含基金名称、交易代码、分级情况、所属类别、保本情况、上市信息、相关机构、投资描述等信息。数据更新频率为不定期。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fund/getFundCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'FundCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'sec_short_name', u'cnSpell', u'establishType', u'establishDate', u'duration', u'expireDate', u'purchaseBeginDate', u'redemptionBeginDate', u'category', u'operationMode', u'isQdii', u'isInnovativeFund', u'isUmbrellaFund', u'isClassFund', u'masterFund', u'sameClassFund', u'isTransFund', u'tickerSymbolBfTrans', u'isGuarFund', u'guarPeriod', u'isSponserFund', u'profitDistributionDesc', u'subscriptionCodeFront', u'subscriptionCodeBack', u'managementCompany', u'custodian', u'perfBenchmark', u'investTarget', u'investField', u'riskReturnCharacter', u'profitDistributionRule', u'fundProfile']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','sec_short_name': 'str','cnSpell': 'str','establishType': 'str','category': 'str','operationMode': 'str','masterFund': 'str','sameClassFund': 'str','tickerSymbolBfTrans': 'str','profitDistributionDesc': 'str','subscriptionCodeFront': 'str','subscriptionCodeBack': 'str','perfBenchmark': 'str','investTarget': 'str','investField': 'str','riskReturnCharacter': 'str','profitDistributionRule': 'str','fundProfile': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundDivmCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取某只货币型或短期理财债券型基金的历史收益情况，包含每万份基金单位当日收益，七日年化收益率等信息。收录了2004年以来的历史数据，数据更新频率为日。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 输入单个证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,secID、ticker至少选择一个
    :param ticker: 输入单个基金代码，如"000001"。,secID、ticker至少选择一个
    :param beginDate: 起始日期，输入格式为“YYYYMMDD”，不输则默认为一年前当日。,可空
    :param endDate: 截止日期,输入格式为“YYYYMMDD”，不输则默认为当日。,可空
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
    requestString.append('/api/fund/getFundDivmCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
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
            api_base.handle_error(csvString, 'FundDivmCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundDivmCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'endDate', u'dailyProfit', u'weeklyYield']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundDividendCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取基金历史分红信息，包含基金的单位分红、累计分红、权益登记日、除息日、分红在投资日、红利发放日等信息。收录了1999年以来的历史数据，数据更新频率为不定期。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fund/getFundDividendCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'FundDividendCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundDividendCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'eventProcessCD', u'distBaseDate', u'nav', u'distNetProfit', u'distAmount', u'dividend', u'regDate', u'exDivDate', u'exDivDateField', u'exDivDateOTC', u'paymentDate', u'paymentDateField', u'paymentDateOTC', u'beneficiary', u'reinvestDate', u'reinvAcctDate', u'reinvRedemDate', u'mmfConvertDate', u'mmfProfitAccumPeriod', u'planChgDesc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','eventProcessCD': 'str','beneficiary': 'str','mmfProfitAccumPeriod': 'str','planChgDesc': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundSplitCCXEGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取基金历史拆分折算信息，包含基金的拆分折算日、折算比例、份额变更登记日等信息。收录了2005年以来的历史数据，数据更新频率为不定期。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fund/getFundSplitCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
        csvString = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
        if csvString is None or len(csvString) == 0 or (csvString[0] == '-' and not api_base.is_no_data_warn(csvString, False)) or csvString[0] == '{':
            api_base.handle_error(csvString, 'FundSplitCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundSplitCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'splitTypeCD', u'splitDate', u'splitRatio', u'chgRegDate', u'sharesBf', u'sharesAf', u'navBf', u'navAf']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','splitTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundAssetsCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取非QDII基金的资产配置历史数据，包含定期报告中披露的各种资产类别及占比情况。收录了1998年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认为一年前今天。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认为今天,可空
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
    requestString.append('/api/fund/getFundAssetsCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
            api_base.handle_error(csvString, 'FundAssetsCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundAssetsCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'assetTypeCD', u'marketValue', u'ratioInTA', u'ratioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','assetTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundAssetsQDIICCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取QDII基金的资产配置历史数据，包含定期报告中披露的各种资产类别及占比情况。收录了2007年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认为一年前今天。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
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
    requestString.append('/api/fund/getFundAssetsQDIICCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
            api_base.handle_error(csvString, 'FundAssetsQDIICCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundAssetsQDIICCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'assetTypeCD', u'marketValue', u'ratioInTA', u'ratioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','assetTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsECCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金所投资的股票组合的历史数据，包含定期报告中披露的股票组合明细，包括主动投资与被动投资等相关数据。收录了1998年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认为一年前今天。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
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
    requestString.append('/api/fund/getFundHoldingsECCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
            api_base.handle_error(csvString, 'FundHoldingsECCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsECCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'investTypeCD', u'serialNumber', u'equitySecID', u'equityTicker', u'equityExchangeCD', u'equityShortName', u'holdVolume', u'marketValue', u'ratioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','investTypeCD': 'str','equitySecID': 'str','equityTicker': 'str','equityExchangeCD': 'str','equityShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsBCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金所投资的债券组合的历史数据，包含定期报告中披露的债券组合明细。收录了2002年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认为去年今天。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
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
    requestString.append('/api/fund/getFundHoldingsBCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
            api_base.handle_error(csvString, 'FundHoldingsBCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsBCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'serialNumber', u'bondSecID', u'bondTicker', u'bondExchangeCD', u'bondShortName', u'holdVolume', u'marketValue', u'ratioInNA', u'isConvertible']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','bondSecID': 'str','bondTicker': 'str','bondExchangeCD': 'str','bondShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsFCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金所投资的目标基金的历史数据，包含ETF联接基金、FOF基金定期报告中披露的相关数据。收录了2007年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认输出最近一年相关数据。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认输出最近一年相关数据。,可空
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
    requestString.append('/api/fund/getFundHoldingsFCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
            api_base.handle_error(csvString, 'FundHoldingsFCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsFCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'serialNumber', u'holdFundSecID', u'holdFundTicker', u'holdFundExchangeCD', u'holdFundShortName', u'holdVolume', u'marketValue', u'ratioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','holdFundSecID': 'str','holdFundTicker': 'str','holdFundExchangeCD': 'str','holdFundShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundETFPRListCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取ETF基金交易日的申赎清单基本信息，包括标的指数名称，上一交易日的现金差额、最小申赎单位净值、单位净值，交易日当日的预估现金差额、最小申赎单位、现金替代比例上限、是否允许申购赎回、是否公布IOPV等信息。收录了2006年以来的历史数据，数据更新频率为日。不输入日期则默认获取近两天的历史数据。
    
    :param secID: 输入单个证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,secID、ticker至少选择一个
    :param ticker: 输入单个基金代码，如"000001"。,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
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
    requestString.append('/api/fund/getFundETFPRListCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(ticker, str) and not isinstance(ticker, unicode):
        ticker = str(ticker)

    requestString.append("&ticker=%s"%(ticker))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'FundETFPRListCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundETFPRListCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'fundShortName', u'underLyingIndex', u'underLyingIndexCode', u'idxShortName', u'preTradeDate', u'cashComp', u'navPerCU', u'nav', u'tradeDate', u'estCashComp', u'maxCashRatio', u'creationUnit', u'ifIOPV', u'ifPurchaseble', u'ifRedeemable']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','fundShortName': 'str','underLyingIndex': 'str','underLyingIndexCode': 'str','idxShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundETFConsCCXEGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取ETF基金每个交易日的跟踪的标的指数成分券清单，包括成分券的代码、简称、股票数量、现金替代溢价比、固定替代金额等信息。收录了2006年以来的历史数据，数据更新频率为日。不输入日期则默认获取近两天的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003"。,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告起始日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
    :param endDate: 报告截止日期，输入"YYYYMMDD"格式。不输入则默认为今天。,可空
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
    requestString.append('/api/fund/getFundETFConsCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'FundETFConsCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundETFConsCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'tradeDate', u'consID', u'consTicker', u'consExchangeCD', u'consName', u'quantity', u'cashSubsSign', u'cashRatio', u'fixedCashAmount']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','consID': 'str','consTicker': 'str','consExchangeCD': 'str','consName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktbBondIdxdCCXEGet(indexID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取交易所债券指数日行情表现，包含证券内部编码,指数交易代码,指数简称,指数英文简称,交易市场,交易日期,昨收盘价,开盘价,最高价,最低价,收盘价,成交金额,成交数量,成交笔数,价格升跌等，历史数据追溯至1990年，每日更新。
    
    :param indexID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'IDX',输入'000001'，可获取到secID'000001.ZICN'后，在此输入'000001.ZICN'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/idx/getMktbBondIdxdCCXE.csv?ispandas=1&') 
    if not isinstance(indexID, str) and not isinstance(indexID, unicode):
        indexID = str(indexID)

    requestString.append("indexID=%s"%(indexID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktbBondIdxdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktbBondIdxdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'indexID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preCloseIndex', u'openIndex', u'highestIndex', u'lowestIndex', u'closeIndex', u'turnoverValue', u'turnovervVol', u'dealAmount', u'chg']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'indexID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktEquIdxdCCXEGet(indexID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取股票指数日行情，包含证券内部编码,指数代码,指数简称,指数英文简称,交易市场,交易日期,昨收盘价,开盘价,最高价,最低价,收盘价,成交金额,成交数量,成交笔数,升跌等，历史追溯至1990年，每日更新。
    
    :param indexID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'idx',输入'000001'，可获取到secID'000001.ZICN'后，在此输入'000001.ZICN'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/idx/getMktEquIdxdCCXE.csv?ispandas=1&') 
    if not isinstance(indexID, str) and not isinstance(indexID, unicode):
        indexID = str(indexID)

    requestString.append("indexID=%s"%(indexID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktEquIdxdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktEquIdxdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'indexID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openIndex', u'highestIndex', u'lowestIndex', u'closeindex', u'turnoverValue', u'turnoverVol', u'dealAmount', u'chg']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'indexID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def IdxECCXEGet(secID = "", ticker = "", corgFullName = "", listStatusCD = "", field = "", pandas = "1"):
    """
    获取股票指数的基本要素信息，包括指数名称、指数代码、指数类型、发布机构、发布日期、基日、基点、加权方式等。
    
    :param secID: 指数展示代码,可通过指数代码和证券类型从DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker、corgFullName至少选择一个
    :param ticker: 指数代码,如:000001 上海证券交易所综合指数,可以是列表,secID、ticker、corgFullName至少选择一个
    :param corgFullName: 编制机构全称，使用模糊匹配，每次只能查询一个发布机构,可以是列表,secID、ticker、corgFullName至少选择一个
    :param listStatusCD: 指数状态,可以是列表,可空
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
    requestString.append('/api/idx/getIdxECCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&corgFullName=")
    if hasattr(corgFullName,'__iter__') and not isinstance(corgFullName, str):
        if len(corgFullName) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = corgFullName
            requestString.append(None)
        else:
            requestString.append(','.join(corgFullName))
    else:
        requestString.append(corgFullName)
    requestString.append("&listStatusCD=")
    if hasattr(listStatusCD,'__iter__') and not isinstance(listStatusCD, str):
        if len(listStatusCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = listStatusCD
            requestString.append(None)
        else:
            requestString.append(','.join(listStatusCD))
    else:
        requestString.append(listStatusCD)
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
            api_base.handle_error(csvString, 'IdxECCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'IdxECCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secFullName', u'secShortName', u'secShortNameEn', u'cnSpell', u'ticker', u'exchangeCD', u'indexPrepObj', u'corgIndexType', u'industryVersionCD', u'industryID', u'industryName', u'corgFullName', u'porgFullName', u'publishDate', u'baseDate', u'basePoint', u'wMethod', u'consNum', u'consAdPeriod', u'listStatusCD', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secFullName': 'str','secShortName': 'str','secShortNameEn': 'str','cnSpell': 'str','ticker': 'str','exchangeCD': 'str','indexPrepObj': 'str','corgIndexType': 'str','industryVersionCD': 'str','industryID': 'str','industryName': 'str','corgFullName': 'str','porgFullName': 'str','publishDate': 'str','baseDate': 'str','wMethod': 'str','consAdPeriod': 'str','listStatusCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def IdxBCCXEGet(secID = "", ticker = "", corgFullName = "", listStatusCD = "", field = "", pandas = "1"):
    """
    获取债券指数的基本要素信息，包括指数名称、指数代码、指数类型、发布机构、发布日期、基日、基点、加权方式等。
    
    :param secID: 指数展示代码,可通过指数代码和证券类型从DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker、corgFullName至少选择一个
    :param ticker: 指数代码,如:000013 上海证券交易所企业债券指数,可以是列表,secID、ticker、corgFullName至少选择一个
    :param corgFullName: 编制机构全称，使用机构全名模糊匹配，每次只能查询一个机构,secID、ticker、corgFullName至少选择一个
    :param listStatusCD: 指数状态,可以是列表,可空
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
    requestString.append('/api/idx/getIdxBCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(corgFullName, str) and not isinstance(corgFullName, unicode):
        corgFullName = str(corgFullName)

    requestString.append("&corgFullName=%s"%(corgFullName))
    requestString.append("&listStatusCD=")
    if hasattr(listStatusCD,'__iter__') and not isinstance(listStatusCD, str):
        if len(listStatusCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = listStatusCD
            requestString.append(None)
        else:
            requestString.append(','.join(listStatusCD))
    else:
        requestString.append(listStatusCD)
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
            api_base.handle_error(csvString, 'IdxBCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'IdxBCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secFullName', u'secShortName', u'secShortNameEn', u'cnSpell', u'ticker', u'exchangeCD', u'indexPrepObj', u'corgIndexType', u'corgFullName', u'porgFullName', u'publishDate', u'baseDate', u'basePoint', u'wMethod', u'consNum', u'consAdPeriod', u'listStatusCD', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secFullName': 'str','secShortName': 'str','secShortNameEn': 'str','cnSpell': 'str','ticker': 'str','exchangeCD': 'str','indexPrepObj': 'str','corgIndexType': 'str','corgFullName': 'str','porgFullName': 'str','publishDate': 'str','baseDate': 'str','wMethod': 'str','consAdPeriod': 'str','listStatusCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def IdxConsECCXEGet(secID = "", ticker = "", isNew = "", field = "", pandas = "1"):
    """
    获取股票指数的成分构成情况，包括指数成分证券名称、成分证券代码、入选日期、剔除日期等。
    
    :param secID: 指数展示代码,可通过指数代码和证券类型从DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker至少选择一个
    :param ticker: 指数代码,如:000001 上海证券交易所综合指数,可以是列表,secID、ticker至少选择一个
    :param isNew: 是否最新:1 是；0 否,可以是列表,可空
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
    requestString.append('/api/idx/getIdxConsECCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&isNew=")
    if hasattr(isNew,'__iter__') and not isinstance(isNew, str):
        if len(isNew) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isNew
            requestString.append(None)
        else:
            requestString.append(','.join(isNew))
    else:
        requestString.append(isNew)
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
            api_base.handle_error(csvString, 'IdxConsECCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'IdxConsECCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secShortName', u'secShortNameEn', u'ticker', u'consID', u'consShortName', u'consShortNameEn', u'consticker', u'consExchangeCD', u'intoDate', u'outDate', u'isNew', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','consID': 'str','consShortName': 'str','consShortNameEn': 'str','consticker': 'str','consExchangeCD': 'str','intoDate': 'str','outDate': 'str','isNew': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def IdxConsBCCXEGet(secID = "", ticker = "", isNew = "", field = "", pandas = "1"):
    """
    获取债券指数的成分构成情况，包括指数成分证券名称、成分证券代码、入选日期、剔除日期等。
    
    :param secID: 指数展示代码,可通过指数代码和证券类型从DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker至少选择一个
    :param ticker: 指数代码,如:000013 上海证券交易所企业债券指数,可以是列表,secID、ticker至少选择一个
    :param isNew: 是否最新:1 是；0 否,可以是列表,可空
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
    requestString.append('/api/idx/getIdxConsBCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&isNew=")
    if hasattr(isNew,'__iter__') and not isinstance(isNew, str):
        if len(isNew) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isNew
            requestString.append(None)
        else:
            requestString.append(','.join(isNew))
    else:
        requestString.append(isNew)
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
            api_base.handle_error(csvString, 'IdxConsBCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'IdxConsBCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secShortName', u'secShortNameEn', u'ticker', u'consID', u'consShortName', u'consShortNameEn', u'consticker', u'consExchangeCD', u'intoDate', u'outDate', u'isNew', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','consID': 'str','consShortName': 'str','consShortNameEn': 'str','consticker': 'str','consExchangeCD': 'str','intoDate': 'str','outDate': 'str','isNew': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBondChiborCCXEGet(startDate = "", endDate = "", muturityCD = "", field = "", pandas = "1"):
    """
    获取银行间拆借交易行情,包括交易日期、拆借代码、拆借期限、开盘利率、最高利率、最低利率、收盘利率、加权平均利率、升降、成交笔数、成交金额、增减等，历史数据追溯至1996年，每日更新。
    
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
    :param muturityCD: 交易期限，1D为隔夜，1M为1个月，1W为7天，1Y为1年，2M为2个月，2W为14天，3M为3个月，3W为21天，4M为4个月，6M为6个月，9M为9个月,可空
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
    requestString.append('/api/moneyMarket/getMktBondChiborCCXE.csv?ispandas=1&') 
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(muturityCD, str) and not isinstance(muturityCD, unicode):
        muturityCD = str(muturityCD)

    requestString.append("&muturityCD=%s"%(muturityCD))
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
            api_base.handle_error(csvString, 'MktBondChiborCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBondChiborCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'tradeDate', u'ticker', u'muturityCD', u'openRate', u'highestRate', u'lowestRate', u'closeRate', u'wRate', u'chg', u'dealAmount', u'turnoverValue', u'chgValue']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','ticker': 'str','muturityCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktBondShiborCCXEGet(partyID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取上海同业拆借利率，利率期限包含隔夜到1年之间，历史追溯到2006年，每日更新。
    
    :param partyID: 报价银行代码，78为国家开发银行股份有限公司，603为上海浦东发展银行股份有限公司，616为华夏银行股份有限公司，629为招商银行股份有限公司，1667为中国建设银行股份有限公司，1867为中国工商银行股份有限公司，1892为兴业银行股份有限公司，1904为南京银行股份有限公司，2035为中国银行股份有限公司，2041为中信银行股份有限公司，2149为北京银行股份有限公司，2274为交通银行股份有限公司，27303为中国农业银行股份有限公司，27523为中国光大银行股份有限公司
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/moneyMarket/getMktBondShiborCCXE.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktBondShiborCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktBondShiborCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'tradeDate', u'partyID', u'askPrice1D', u'bidPrice1D', u'askPrice1W', u'bidPrice1W', u'askPrice2W', u'bidPrice2W', u'askPrice2M', u'bidPrice2M', u'askPrice3M', u'bidPrice3M', u'askPrice6M', u'bidPrice6M', u'askPrice9M', u'bidPrice9M', u'askPrice1Y', u'bidPrice1Y']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktSwapdCCXEGet(endDate = "", refRatePer = "", maturityCD = "", field = "", pandas = "1"):
    """
    获取利率互换日行情，包含截止日期,参考利率,互换期限,固定利率,名义本金等，历史数据追溯至2008年，每日更新。
    
    :param endDate: 截止日期，输入格式为yyyymmdd,可空
    :param refRatePer: FD6M为半年期整存整取定期存款利率,FD1Y为一年期整存整取定期存款利率,LOAN6M为六个月以内(含六个月)贷款利率,LOAN6M1Y为六个月至一年期(含一年)贷款利率,LOAN1Y3Y为一至三年期(含三年)贷款利率,LOAN3Y5Y为三至五年期(含五年)贷款利率,LOAN5Y为五年期以上贷款利率,FR007为7天回购定盘利率(FR007),LIBOR3M为LIBOR(3个月)SHIBOR3M10D为SHIBOR(3个月)10日均值,SHIBORON为SHIBOR(隔夜),SHIBOR1W为SHIBOR(1 周),LIBOR为LIBOR,LPR1Y为1年期贷款基础利率,R007为银行间市场7天回购利率,可空
    :param maturityCD: 1.5Y为1.5年，10Y为10年，1M为1个月，1W为1周，1Y为1年，2W为2周，3M为3个月，3W为3周，3Y为3年，4Y为4年，5Y为5年，6M为6个月，6Y为6年，7Y为7年，8Y为8年，9M为9个月，9Y为9年,可空
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
    requestString.append('/api/moneyMarket/getMktSwapdCCXE.csv?ispandas=1&') 
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("endDate=%s"%(endDate))
    if not isinstance(refRatePer, str) and not isinstance(refRatePer, unicode):
        refRatePer = str(refRatePer)

    requestString.append("&refRatePer=%s"%(refRatePer))
    if not isinstance(maturityCD, str) and not isinstance(maturityCD, unicode):
        maturityCD = str(maturityCD)

    requestString.append("&maturityCD=%s"%(maturityCD))
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
            api_base.handle_error(csvString, 'MktSwapdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktSwapdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'endDate', u'refRatePer', u'maturityCD', u'fixRate', u'notiPrinc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'endDate': 'str','refRatePer': 'str','maturityCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktSwapmCCXEGet(endDate = "", refRatePer = "", maturityCD = "", field = "", pandas = "1"):
    """
    获取利率互换月行情统计，包含截止日期,参考利率,互换期限,固定利率,成交笔数,名义本金等，历史数据追溯至2006年，每月更新。
    
    :param endDate: 截止日期，输入格式为yyyymmdd,可空
    :param refRatePer: FD6M为半年期整存整取定期存款利率,FD1Y为一年期整存整取定期存款利率,LOAN6M为六个月以内(含六个月)贷款利率,LOAN6M1Y为六个月至一年期(含一年)贷款利率,LOAN1Y3Y为一至三年期(含三年)贷款利率,LOAN3Y5Y为三至五年期(含五年)贷款利率,LOAN5Y为五年期以上贷款利率,FR007为7天回购定盘利率(FR007),LIBOR3M为LIBOR(3个月)SHIBOR3M10D为SHIBOR(3个月)10日均值,SHIBORON为SHIBOR(隔夜),SHIBOR1W为SHIBOR(1 周),LIBOR为LIBOR,LPR1Y为1年期贷款基础利率,R007为银行间市场7天回购利率,可空
    :param maturityCD: 1.5Y为1.5年，10Y为10年，1M为1个月，1W为1周，1Y为1年，2W为2周，3M为3个月，3W为3周，3Y为3年，4Y为4年，5Y为5年，6M为6个月，6Y为6年，7Y为7年，8Y为8年，9M为9个月，9Y为9年,可空
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
    requestString.append('/api/moneyMarket/getMktSwapmCCXE.csv?ispandas=1&') 
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("endDate=%s"%(endDate))
    if not isinstance(refRatePer, str) and not isinstance(refRatePer, unicode):
        refRatePer = str(refRatePer)

    requestString.append("&refRatePer=%s"%(refRatePer))
    if not isinstance(maturityCD, str) and not isinstance(maturityCD, unicode):
        maturityCD = str(maturityCD)

    requestString.append("&maturityCD=%s"%(maturityCD))
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
            api_base.handle_error(csvString, 'MktSwapmCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktSwapmCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'endDate', u'refRatePer', u'maturityCD', u'fixRate', u'dealAmount', u'notiPrinc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'endDate': 'str','refRatePer': 'str','maturityCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtBSCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间资产负债表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期,起始时间,如‘20121231’,可空
    :param endDateRep: 报告的会计期间截止日期,结束时间,如‘20131231’,可空
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
    requestString.append('/api/fundamental/getFdmtBSCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(beginDateRep, str) and not isinstance(beginDateRep, unicode):
        beginDateRep = str(beginDateRep)

    requestString.append("&beginDateRep=%s"%(beginDateRep))
    if not isinstance(endDateRep, str) and not isinstance(endDateRep, unicode):
        endDateRep = str(endDateRep)

    requestString.append("&endDateRep=%s"%(endDateRep))
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
            api_base.handle_error(csvString, 'FdmtBSCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtBSCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'publishDate', u'endDateRep', u'endDate', u'mergedFlag', u'reportType', u'busiTypePar', u'cashCEquiv', u'clientDepos', u'tradingFA', u'notesReceiv', u'divReceiv', u'intReceiv', u'AR', u'othReceiv', u'prepayment', u'inventories', u'BBA', u'deferExp', u'NCAWithin1Y', u'settProv', u'loanToOthBankFI', u'premiumReceiv', u'reinsurReceiv', u'reinsurReserReceiv', u'purResaleFA', u'othCA', u'CAE', u'CAA', u'TCA', u'availForSaleFA', u'htmInvest', u'investRealEstate', u'LTEquityInvest', u'LTReceiv', u'fixedAssets', u'constMaterials', u'CIP', u'fixedAssetsDisp', u'producBiolAssets', u'oilAndGasAssets', u'intanAssets', u'transacSeatFee', u'RD', u'goodwill', u'LTAmorExp', u'deferTaxAssets', u'disburLA', u'othNCA', u'NCAE', u'NCAA', u'TNCA', u'investAsReceiv', u'discountAssets', u'clientProv', u'deposInOthBFI', u'preciMetals', u'derivAssets', u'subrogRecoReceiv', u'RRReinsUnePrem', u'RRReinsOutstdCla', u'RRReinsLinsLiab', u'RRReinsLthinsLiab', u'PHPledgeLoans', u'fixedTermDepos', u'refundDepos', u'capitalVicaBusi', u'refundCapDepos', u'indepAccAssets', u'othAssets', u'AE', u'AA', u'TAssets', u'STBorr', u'pledgeBorr', u'tradingFL', u'notesPayable', u'AP', u'STBondPayable', u'advanceReceipts', u'payrollPayable', u'divPayable', u'taxesPayable', u'intPayable', u'othPayable', u'accrExp', u'deferIncomeST', u'CBBorr', u'deposPrepayment', u'loanFrOthBankFI', u'soldForRepurFA', u'commisPayable', u'reinsurPayable', u'insurReser', u'fundsSecTradAgen', u'fundsSecUndwAgen', u'NCLWithin1Y', u'othCl', u'CLE', u'CLA', u'TCL', u'LTBorr', u'bondPayable', u'LTPayable', u'specificPayables', u'estimatedLiab', u'deferTaxLiab', u'deferIncome', u'othNCL', u'NCLE', u'NCLA', u'TNCL', u'deposFrOthBFI', u'derivLiab', u'liabVicaBusi', u'depos', u'deposReceiv', u'premReceivAdva', u'indemAccPayable', u'policyDivPayable', u'PHInvest', u'reserUnePrem', u'reserOutstdClaims', u'reserLinsLiab', u'reserLthinsLiab', u'indeptAccLiab', u'othLiab', u'LE', u'LA', u'TLiab', u'paidInCapital', u'capitalReser', u'treasuryShare', u'specialReser', u'surplusReser', u'retainedEarnings', u'ordinRiskReser', u'forexDiffer', u'unInvestlLoss', u'othReser', u'SEE', u'SEA', u'TEquityAttrP', u'minorityInt', u'othEffectSe', u'TShEquity', u'LEE', u'LEA', u'TLiabEquity', u'specFieldRemark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDateRep': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str','busiTypePar': 'str','specFieldRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtISCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间利润表的累计数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/fundamental/getFdmtISCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'FdmtISCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtISCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'publishDate', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'reportType', u'busiTypePar', u'TRevenue', u'revenue', u'NIntIncome', u'intIncome', u'intExp', u'NCommisIncome', u'commisIncome', u'commisExp', u'NSecTaIncome', u'NUndwrtSecIncome', u'NTrustIncome', u'premEarned', u'grossPremWrit', u'reinsIncome', u'reinsur', u'unePremReser', u'othRevenue', u'specOR', u'AOR', u'TCOGS', u'payout', u'premRefund', u'compensPayout', u'compensPayoutRefu', u'reserInsurLiab', u'insurLiabReserRefu', u'policyDivPayt', u'reinsurExp', u'genlAdminExp', u'reinsCostRefund', u'insurCommisExp', u'bizTaxSurchg', u'othCost', u'COGS', u'sellExp', u'adminExp', u'finanExp', u'assetsImpairLoss', u'specTOC', u'ATOC', u'othNetRevenue', u'FValueChgGain', u'investIncome', u'AJInvestIncome', u'forexGain', u'othEffectOP', u'AEffectOP', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'NCADisploss', u'othEffectTP', u'AEffectTP', u'TProfit', u'incomeTax', u'unInvestLoss', u'othEffectNP', u'AEffectNP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'othEffectNPP', u'AEffectNPP', u'othComprIncome', u'AEffectCI', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS', u'AEffectPCI', u'basicEPS', u'dilutedEPS', u'specFieldRemark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDateRep': 'str','startDate': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str','busiTypePar': 'str','specFieldRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtISQCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间利润表的单季数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/fundamental/getFdmtISQCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'FdmtISQCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtISQCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'publishDate', u'endDate', u'mergedFlag', u'reportType', u'TRevenue', u'revenue', u'NIntIncome', u'intIncome', u'intExp', u'NCommisIncome', u'commisIncome', u'commisExp', u'NSecTaIncome', u'NUndwrtSecIncome', u'NTrustIncome', u'premEarned', u'grossPremWrit', u'reinsIncome', u'reinsur', u'unePremReser', u'forexGain', u'othRevenue', u'specOR', u'AOR', u'TCOGS', u'COGS', u'sellExp', u'adminExp', u'finanExp', u'payout', u'premRefund', u'compensPayout', u'compensPayoutRefu', u'reserInsurLiab', u'insurLiabReserRefu', u'policyDivPayt', u'reinsurExp', u'insurCommisExp', u'bizTaxSurchg', u'genlAdminExp', u'reinsCostRefund', u'assetsImpairLoss', u'othCost', u'specTOC', u'ATOC', u'investIncome', u'AJInvestIncome', u'FValueChgGain', u'othEffectOP', u'AEffectOP', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'NCADisploss', u'othEffectTP', u'AEffectTP', u'TProfit', u'incomeTax', u'unInvestLoss', u'othEffectNP', u'AEffectNP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'othComprIncome', u'AEffectCI', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS', u'AEffectPCI']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtCFCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间现金流量表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期,起始时间,如‘20121231’,可空
    :param endDateRep: 报告的会计期间截止日期,结束时间,如‘20131231’,可空
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
    requestString.append('/api/fundamental/getFdmtCFCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(beginDateRep, str) and not isinstance(beginDateRep, unicode):
        beginDateRep = str(beginDateRep)

    requestString.append("&beginDateRep=%s"%(beginDateRep))
    if not isinstance(endDateRep, str) and not isinstance(endDateRep, unicode):
        endDateRep = str(endDateRep)

    requestString.append("&endDateRep=%s"%(endDateRep))
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
            api_base.handle_error(csvString, 'FdmtCFCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtCFCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'publishDate', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'reportType', u'busiTypePar', u'CFrSaleGS', u'refundOfTax', u'NDeposIncrCFI', u'NIncrBorrFrCB', u'NIncBorrOthFI', u'drawBackLoansC', u'IFCCashIncr', u'NIncDispTradFA', u'NIncDispFAFS', u'NCapIncrRepur', u'premFrOrigContr', u'NReinsurPrem', u'NIncPhDeposInv', u'CFrOthOperateA', u'specOCIF', u'AOCIF', u'CInfFrOperateA', u'CPaidGS', u'CPaidToForEmpl', u'CPaidForTaxes', u'NIncDisburOfLA', u'netIncrDeposInFI', u'NIncrLoansToOthFI', u'CPaidIFC', u'origContrCIndem', u'NForReinsurPrem', u'CPaidPolDiv', u'CPaidForOthOpA', u'specOCOF', u'AOCOF', u'COutfOperateA', u'ANOCF', u'NCFOperateA', u'procSellInvest', u'gainInvest', u'dispFixAssetsOth', u'NDispSubsOthBizC', u'CFrOthInvestA', u'specCIF', u'ACIF', u'CInfFrInvestA', u'purFixAssetsOth', u'CPaidInvest', u'NCPaidAcquis', u'NIncrPledgeLoan', u'CPaidOthInvestA', u'specCOF', u'ACOF', u'COutfFrInvestA', u'ANICF', u'NCFFrInvestA', u'CFrCapContr', u'CFrMinoSSubs', u'CFrIssueBond', u'NIncPhDepos', u'CFrBorr', u'CFrOthFinanA', u'specFCIF', u'AFCIF', u'CInfFrFinanA', u'CPaidForDebts', u'CPaidDivProfInt', u'divProfSubsMinoS', u'CPaidOthFinanA', u'specFCOF', u'AFCOF', u'COutfFrFinanA', u'ANFCF', u'NCFFrFinanA', u'forexEffects', u'othEffectCE', u'ACE', u'NChangeInCash', u'NCEBegBal', u'othEffectCEI', u'ACEI', u'NCEEndBal', u'NIncome', u'assetsImpairLoss', u'FADepr', u'intanAssetsAmor', u'LTAmorExpAmor', u'amorExpDecr', u'accrExpIncr', u'dispFAOthLoss', u'FAWritOff', u'FValueChgLoss', u'finanExp', u'invLoss', u'deferTADecr', u'deferTLIncr', u'invenDecr', u'operReceiDecr', u'operPayaIncr', u'unInvestLoss', u'other', u'specNOCF1', u'ANOCF1', u'NCFOperateANotes', u'contrANOCF', u'convDebtCapi', u'convBonds1Y', u'finanLeaFA', u'CEndBal', u'CBegBal', u'CEEndBal', u'CEBegBal', u'specC', u'AC', u'NChangeInCashNotes', u'contrANC', u'specFieldRemark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDateRep': 'str','startDate': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str','busiTypePar': 'str','specFieldRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondFdmtBSCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了债券发行人定期报告中各个期间资产负债表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'081601.XIBE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'081601',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/fundamental/getBondFdmtBSCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'BondFdmtBSCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondFdmtBSCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'partyID', u'secShortName', u'publishDate', u'endDate', u'mergedFlag', u'reportType', u'cashCEquiv', u'clientDepos', u'tradingFA', u'notesReceiv', u'divReceiv', u'intReceiv', u'AR', u'othReceiv', u'prepayment', u'inventories', u'BBA', u'deferExp', u'NCAWithin1Y', u'settProv', u'loanToOthBankFI', u'premiumReceiv', u'reinsurReceiv', u'reinsurReserReceiv', u'purResaleFA', u'othCA', u'CAE', u'CAA', u'TCA', u'availForSaleFA', u'htmInvest', u'investRealEstate', u'LTEquityInvest', u'LTReceiv', u'fixedAssets', u'constMaterials', u'CIP', u'fixedAssetsDisp', u'producBiolAssets', u'oilAndGasAssets', u'intanAssets', u'transacSeatFee', u'RD', u'goodwill', u'LTAmorExp', u'deferTaxAssets', u'disburLA', u'othNCA', u'NCAE', u'NCAA', u'TNCA', u'investAsReceiv', u'discountAssets', u'clientProv', u'deposInOthBFI', u'preciMetals', u'derivAssets', u'subrogRecoReceiv', u'RRReinsUnePrem', u'RRReinsOutstdCla', u'RRReinsLinsLiab', u'RRReinsLthinsLiab', u'PHPledgeLoans', u'fixedTermDepos', u'refundDepos', u'capitalVicaBusi', u'refundCapDepos', u'indepAccAssets', u'othAssets', u'AE', u'AA', u'TAssets', u'STBorr', u'pledgeBorr', u'tradingFL', u'notesPayable', u'AP', u'STBondPayable', u'advanceReceipts', u'payrollPayable', u'divPayable', u'taxesPayable', u'intPayable', u'othPayable', u'accrExp', u'deferIncomeST', u'CBBorr', u'deposPrepayment', u'loanFrOthBankFI', u'soldForRepurFA', u'commisPayable', u'reinsurPayable', u'insurReser', u'fundsSecTradAgen', u'fundsSecUndwAgen', u'NCLWithin1Y', u'othCl', u'CLE', u'CLA', u'TCL', u'LTBorr', u'bondPayable', u'LTPayable', u'specificPayables', u'estimatedLiab', u'deferTaxLiab', u'deferIncome', u'othNCL', u'NCLE', u'NCLA', u'TNCL', u'deposFrOthBFI', u'derivLiab', u'liabVicaBusi', u'depos', u'deposReceiv', u'premReceivAdva', u'indemAccPayable', u'policyDivPayable', u'PHInvest', u'reserUnePrem', u'reserOutstdClaims', u'reserLinsLiab', u'reserLthinsLiab', u'indeptAccLiab', u'othLiab', u'LE', u'LA', u'TLiab', u'paidInCapital', u'capitalReser', u'treasuryShare', u'specialReser', u'surplusReser', u'retainedEarnings', u'ordinRiskReser', u'forexDiffer', u'unInvestlLoss', u'othReser', u'SEE', u'SEA', u'TEquityAttrP', u'minorityInt', u'othEffectSe', u'TShEquity', u'LEE', u'LEA', u'TLiabEquity', u'specFieldRemark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str','specFieldRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondFdmtISCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了债券发行人定期报告中各个期间利润表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'081601.XIBE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'081601',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期,起始时间,如‘20121231’,可空
    :param endDateRep: 报告的会计期间截止日期,结束时间,如‘20131231’,可空
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
    requestString.append('/api/fundamental/getBondFdmtISCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(beginDateRep, str) and not isinstance(beginDateRep, unicode):
        beginDateRep = str(beginDateRep)

    requestString.append("&beginDateRep=%s"%(beginDateRep))
    if not isinstance(endDateRep, str) and not isinstance(endDateRep, unicode):
        endDateRep = str(endDateRep)

    requestString.append("&endDateRep=%s"%(endDateRep))
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
            api_base.handle_error(csvString, 'BondFdmtISCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondFdmtISCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'partyID', u'secShortName', u'publishDate', u'startDate', u'endDate', u'mergedFlag', u'reportType', u'TRevenue', u'revenue', u'NIntIncome', u'intIncome', u'intExp', u'NCommisIncome', u'commisIncome', u'commisExp', u'NSecTaIncome', u'NUndwrtSecIncome', u'NTrustIncome', u'premEarned', u'grossPremWrit', u'reinsIncome', u'reinsur', u'unePremReser', u'othRevenue', u'specOR', u'AOR', u'TCOGS', u'payout', u'premRefund', u'compensPayout', u'compensPayoutRefu', u'reserInsurLiab', u'insurLiabReserRefu', u'policyDivPayt', u'reinsurExp', u'genlAdminExp', u'reinsCostRefund', u'insurCommisExp', u'bizTaxSurchg', u'othCost', u'COGS', u'sellExp', u'adminExp', u'finanExp', u'assetsImpairLoss', u'specTOC', u'ATOC', u'othNetRevenue', u'FValueChgGain', u'investIncome', u'AJInvestIncome', u'forexGain', u'othEffectOP', u'AEffectOP', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'NCADisploss', u'othEffectTP', u'AEffectTP', u'TProfit', u'incomeTax', u'unInvestLoss', u'othEffectNP', u'AEffectNP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'othEffectNPP', u'AEffectNPP', u'othComprIncome', u'AEffectCI', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS', u'AEffectPCI', u'basicEPS', u'dilutedEPS', u'specFieldRemark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','startDate': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str','specFieldRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondFdmtCFCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了债券发行人定期报告中各个期间现金流量表的数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、如果上市公司对外财务报表进行更正，调整，该表展示最新调整数据； 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'081601.XIBE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'081601',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，默认为1年前当前日期，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期,起始时间,如‘20121231’,可空
    :param endDateRep: 报告的会计期间截止日期,结束时间,如‘20131231’,可空
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
    requestString.append('/api/fundamental/getBondFdmtCFCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
    if not isinstance(beginDateRep, str) and not isinstance(beginDateRep, unicode):
        beginDateRep = str(beginDateRep)

    requestString.append("&beginDateRep=%s"%(beginDateRep))
    if not isinstance(endDateRep, str) and not isinstance(endDateRep, unicode):
        endDateRep = str(endDateRep)

    requestString.append("&endDateRep=%s"%(endDateRep))
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
            api_base.handle_error(csvString, 'BondFdmtCFCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondFdmtCFCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'partyID', u'secShortName', u'publishDate', u'startDate', u'endDate', u'mergedFlag', u'reportType', u'CFrSaleGS', u'refundOfTax', u'NDeposIncrCFI', u'NIncrBorrFrCB', u'NIncBorrOthFI', u'drawBackLoansC', u'IFCCashIncr', u'NIncDispTradFA', u'NIncDispFAFS', u'NCapIncrRepur', u'premFrOrigContr', u'NReinsurPrem', u'NIncPhDeposInv', u'CFrOthOperateA', u'specOCIF', u'AOCIF', u'CInfFrOperateA', u'CPaidGS', u'CPaidToForEmpl', u'CPaidForTaxes', u'NIncDisburOfLA', u'netIncrDeposInFI', u'NIncrLoansToOthFI', u'CPaidIFC', u'origContrCIndem', u'NForReinsurPrem', u'CPaidPolDiv', u'CPaidForOthOpA', u'specOCOF', u'AOCOF', u'COutfOperateA', u'ANOCF', u'NCFOperateA', u'procSellInvest', u'gainInvest', u'dispFixAssetsOth', u'NDispSubsOthBizC', u'CFrOthInvestA', u'specCIF', u'ACIF', u'CInfFrInvestA', u'purFixAssetsOth', u'CPaidInvest', u'NCPaidAcquis', u'NIncrPledgeLoan', u'CPaidOthInvestA', u'specCOF', u'ACOF', u'COutfFrInvestA', u'ANICF', u'NCFFrInvestA', u'CFrCapContr', u'CFrMinoSSubs', u'CFrIssueBond', u'NIncPhDepos', u'CFrBorr', u'CFrOthFinanA', u'specFCIF', u'AFCIF', u'CInfFrFinanA', u'CPaidForDebts', u'CPaidDivProfInt', u'divProfSubsMinoS', u'CPaidOthFinanA', u'specFCOF', u'AFCOF', u'COutfFrFinanA', u'ANFCF', u'NCFFrFinanA', u'forexEffects', u'othEffectCE', u'ACE', u'NChangeInCash', u'NCEBegBal', u'othEffectCEI', u'ACEI', u'NCEEndBal', u'NIncome', u'assetsImpairLoss', u'FADepr', u'intanAssetsAmor', u'LTAmorExpAmor', u'amorExpDecr', u'accrExpIncr', u'dispFAOthLoss', u'FAWritOff', u'FValueChgLoss', u'finanExp', u'invLoss', u'deferTADecr', u'deferTLIncr', u'invenDecr', u'operReceiDecr', u'operPayaIncr', u'unInvestLoss', u'other', u'specNOCF1', u'ANOCF1', u'NCFOperateANotes', u'contrANOCF', u'convDebtCapi', u'convBonds1Y', u'finanLeaFA', u'CEndBal', u'CBegBal', u'CEEndBal', u'CEBegBal', u'specC', u'AC', u'NChangeInCashNotes', u'contrANC', u'specFieldRemark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','startDate': 'str','endDate': 'str','mergedFlag': 'str','reportType': 'str','specFieldRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtEECCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取上市公司披露的业绩快报中的主要财务指标等其他数据，包括本期，去年同期，及本期与期初数值同比数据。（若上市公司同时发行债券等其他证券，也可通过其他证券代码查询快报数据）
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'600000',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/fundamental/getFdmtEECCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'FdmtEECCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtEECCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'publishDate', u'endDate', u'mergedFlag', u'perMarkCD', u'revenue', u'primeOperRev', u'grossProfit', u'operateProfit', u'TProfit', u'NINCOMEAttrP', u'NINCOMECut', u'NCFOperA', u'basicEPS', u'EPSDilu', u'EPSCut', u'EPSCutW', u'ROE', u'ROEW', u'ROECut', u'ROECutW', u'NCFOperAPS', u'TAssets', u'TEquityAttrP', u'paidInCapital', u'NAssetPS', u'revenueLY', u'primeOperRevLY', u'grossProfitLY', u'operProfitLY', u'TProfitLY', u'NIncomeAttrPLY', u'NIncomeCutLY', u'NCFOperALY', u'basicEPSLY', u'EPSDiluLY', u'EPSCutLY', u'EPSCutWLY', u'ROELY', u'ROEWLY', u'ROECutLY', u'ROECutWLY', u'NCFOperAPSLY', u'TAssetsLY', u'TEquityAttrPLY', u'NAssetPSLY', u'revenueYOY', u'primeOperRevYOY', u'grossProfitYOY', u'operProfitYOY', u'TProfitYOY', u'NIncomeAttrPYOY', u'NIncomeCutYOY', u'NCFOperAYOY', u'basicEPSYOY', u'EPSDiluYOY', u'EPSCutYOY', u'EPSCutWYOY', u'ROEFlu', u'ROEWFlu', u'ROECutFlu', u'ROECutWFlu', u'NCFOperAPSYOY', u'TAssetsYOY', u'TEquityAttrPYOY', u'NAssetPSYOY']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDate': 'str','mergedFlag': 'str','perMarkCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtEFCCXEGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取上市公司披露的公告中的预期下一报告期收入、净利润、基本每股收益及其幅度变化等数据。（若上市公司同时发行债券等其他证券，也可通过其他证券代码查询预告数据）
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000005',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/fundamental/getFdmtEFCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
    if not isinstance(beginDate, str) and not isinstance(beginDate, unicode):
        beginDate = str(beginDate)

    requestString.append("&beginDate=%s"%(beginDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'FdmtEFCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtEFCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'publishDate', u'endDate', u'forecastObjCD', u'forecastType', u'pubTime', u'expRevLL', u'expRevUPL', u'RevLast', u'revChgrLL', u'revChgrUPL', u'expnIncAPLL', u'expnIncAPUPL', u'incAPLast', u'NIncAPChgrLL', u'NIncAPChgrUPL', u'expnIncomeLL', u'expnIncomeUPL', u'incomeLast', u'NIncomeChgrLL', u'NIncomeChgrUPL', u'expEPSLL', u'expEPSUPL', u'EPSLast', u'expEPSChgrLL', u'expEPSChgrUPL', u'ifAuditedCD', u'forecastCont']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','publishDate': 'str','endDate': 'str','forecastObjCD': 'str','forecastType': 'str','ifAuditedCD': 'str','forecastCont': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktCCFXifFutdCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取中金所股指期货合约日行情，其中成交量、成交额、持仓量均为单边计算，包含证券内部编码,期货合约代码,合约简称,合约英文简称,交易市场,交易日期,合约交易标的,开盘价,最高价,最低价,结算价,收盘价,成交数量,成交金额,持仓量等，历史追溯至2010年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'fu',输入'IF1503'，可获取到secID'IF1503.CCFX'后，在此输入'IF1503.CCFX'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/future/getMktCCFXifFutdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktCCFXifFutdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktCCFXifFutdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'contractObject', u'openPrice', u'highestPrice', u'lowestPrice', u'settlPrice', u'closePrice', u'turnoverVol', u'turnoverValue', u'openInt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktCCFXtfFutdCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取中金所国债期货合约日行情，其中成交量、持仓量按单边计算，包含证券内部编码,期货合约代码,合约简称,合约英文简称,交易市场,交易日期,合约交易标的,开盘价,最高价,最低价,结算价,收盘价,成交数量,成交金额,持仓量等，历史追溯至2013年9月6日，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'fu',输入'TF1503'，可获取到secID'TF1503.CCFX'后，在此输入'TF1503.CCFX'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/future/getMktCCFXtfFutdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktCCFXtfFutdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktCCFXtfFutdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'contractObject', u'openPrice', u'highestPrice', u'lowestPrice', u'settlPrice', u'closePrice', u'turnoverVol', u'turnoverValue', u'openInt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktXDCEFutdCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取存储大连商品期货交易所期货合约日行情数据，成交量、持仓量、成交额按双边计算，包含证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,品种代码,开盘价,最高价,最低价,结算价,收盘价,成交量,成交金额,持仓量等，历史数据追溯至2009年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'fu',输入'A1505'，可获取到secID'A1505.XDCE'后，在此输入'A1505.XDCE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/future/getMktXDCEFutdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktXDCEFutdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktXDCEFutdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'contractObject', u'openPrice', u'hightestPrice', u'lowestPrice', u'settlPrice', u'closePrice', u'turnoverVol', u'turnoverValue', u'openInt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktXSGEFutdCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取存储上海期货交易所期货合约日行情数据，成交量、持仓量均按双边计算，包括证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,品种代码,开盘价,最高价,最低价,结算价,收盘价,成交量,持仓量等，历史数据追溯至2010年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'fu',输入'CU1505'，可获取到secID'CU1505.XSGE'后，在此输入'CU1505.XSGE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/future/getMktXSGEFutdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktXSGEFutdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktXSGEFutdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'contractObject', u'openPrice', u'hightestPrice', u'lowestPrice', u'settlPrice', u'closePrice', u'turnoverVol', u'openInt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktXZCEFutdCCXEGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取郑州商品交易所期货合约行情，包含证券内部编码,证券代码,证券简称,证券英文简称,交易市场,交易日期,品种代码,开盘价,最高价,最低价,结算价,收盘价,成交量,成交金额,持仓量等，历史追溯至2007年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'fu',输入'CF505'，可获取到secID'CF505.XZCE'后，在此输入'CF505.XZCE'
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param endDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/future/getMktXZCEFutdCCXE.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    if not isinstance(startDate, str) and not isinstance(startDate, unicode):
        startDate = str(startDate)

    requestString.append("&startDate=%s"%(startDate))
    if not isinstance(endDate, str) and not isinstance(endDate, unicode):
        endDate = str(endDate)

    requestString.append("&endDate=%s"%(endDate))
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
            api_base.handle_error(csvString, 'MktXZCEFutdCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktXZCEFutdCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'secShortNameEN', u'exchangeCD', u'tradeDate', u'contractObject', u'openPrice', u'hightestPrice', u'lowestPrice', u'settlPrice', u'closePrice', u'turnoverVol', u'turnoverValue', u'openInt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','secShortNameEN': 'str','exchangeCD': 'str','tradeDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuCfCCXEGet(secID = "", ticker = "", exchangeCD = "", varUniCode = "", contractStatus = "", field = "", pandas = "1"):
    """
    获取商品期货合约的基本要素信息，包括合约名称、合约代码、合约交易品种、上市日期、最后交易日、交割日期、交割结算价、合约状态等。
    
    :param secID: 合约展示代码,可通过合约代码和证券类型在DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param ticker: 合约代码,如:cu1509 沪铜1509合约,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param exchangeCD: 交易所,如:XDCE 大连商品交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param varUniCode: 品种编码,与DataAPI.CCXE.FutuVarcfCCXEGet关联,可获取期货品种基本信息,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param contractStatus: 合约状态:L 上市；DE 已退市；UN 未上市。对应DataAPI.SysCodeGet.codeTypeID=10005。,可以是列表,可空
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
    requestString.append('/api/future/getFutuCfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&varUniCode=")
    if hasattr(varUniCode,'__iter__') and not isinstance(varUniCode, str):
        if len(varUniCode) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = varUniCode
            requestString.append(None)
        else:
            requestString.append(','.join(varUniCode))
    else:
        requestString.append(varUniCode)
    requestString.append("&contractStatus=")
    if hasattr(contractStatus,'__iter__') and not isinstance(contractStatus, str):
        if len(contractStatus) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = contractStatus
            requestString.append(None)
        else:
            requestString.append(','.join(contractStatus))
    else:
        requestString.append(contractStatus)
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
            api_base.handle_error(csvString, 'FutuCfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuCfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secFullName', u'secFullNameEn', u'secShortName', u'secShortNameEn', u'cnSpell', u'ticker', u'exchangeCD', u'isin', u'varUniCode', u'varName', u'varCode', u'listDate', u'deliYear', u'deliMonth', u'lastTradeDate', u'deliDate', u'lastDeliDate', u'deliPriceMethod', u'deliSettPrice', u'contractStatus', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secFullName': 'str','secFullNameEn': 'str','secShortName': 'str','secShortNameEn': 'str','cnSpell': 'str','ticker': 'str','exchangeCD': 'str','isin': 'str','varUniCode': 'str','varName': 'str','varCode': 'str','listDate': 'str','deliYear': 'str','deliMonth': 'str','lastTradeDate': 'str','deliDate': 'str','lastDeliDate': 'str','deliPriceMethod': 'str','contractStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuIfCCXEGet(secID = "", ticker = "", exchangeCD = "", varUniCode = "", contractStatus = "", field = "", pandas = "1"):
    """
    获取股指期货合约的基本要素信息，包括合约名称、合约代码、合约交易品种、上市日期、最后交易日、交割日期、交割结算价、合约状态等。
    
    :param secID: 合约展示代码,可通过合约代码和证券类型在DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param ticker: 合约代码,如:IF1412 沪深300股指期货1412合约,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param exchangeCD: 交易所,如:CCFX 中国金融期货交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param varUniCode: 品种编码,与getFutuVarifCCXE关联,可获取期货品种基本信息,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param contractStatus: 合约状态:L 上市；DE 已退市；UN 未上市。对应DataAPI.SysCodeGet.codeTypeID=10005。,可以是列表,可空
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
    requestString.append('/api/future/getFutuIfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&varUniCode=")
    if hasattr(varUniCode,'__iter__') and not isinstance(varUniCode, str):
        if len(varUniCode) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = varUniCode
            requestString.append(None)
        else:
            requestString.append(','.join(varUniCode))
    else:
        requestString.append(varUniCode)
    requestString.append("&contractStatus=")
    if hasattr(contractStatus,'__iter__') and not isinstance(contractStatus, str):
        if len(contractStatus) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = contractStatus
            requestString.append(None)
        else:
            requestString.append(','.join(contractStatus))
    else:
        requestString.append(contractStatus)
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
            api_base.handle_error(csvString, 'FutuIfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuIfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secFullName', u'secFullNameEn', u'secShortName', u'secShortNameEn', u'cnSpell', u'ticker', u'exchangeCD', u'varUniCode', u'varName', u'varCode', u'listDate', u'deliYear', u'deliMonth', u'lastTradeDate', u'deliDate', u'lastDeliDate', u'deliPriceMethod', u'deliSettPrice', u'contractStatus', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secFullName': 'str','secFullNameEn': 'str','secShortName': 'str','secShortNameEn': 'str','cnSpell': 'str','ticker': 'str','exchangeCD': 'str','varUniCode': 'str','varName': 'str','varCode': 'str','listDate': 'str','deliYear': 'str','deliMonth': 'str','lastTradeDate': 'str','deliDate': 'str','lastDeliDate': 'str','deliPriceMethod': 'str','contractStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuIrfCCXEGet(secID = "", ticker = "", exchangeCD = "", varUniCode = "", contractStatus = "", field = "", pandas = "1"):
    """
    获取国债期货合约的基本要素信息，包括合约名称、合约代码、合约交易品种、上市日期、最后交易日、交割日期、交割结算价、合约状态等。
    
    :param secID: 合约展示代码,可通过合约代码和证券类型在DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param ticker: 合约代码,如:TF1412 票面利率3%的5年期名义中期国债期货1412合约,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param exchangeCD: 交易所,如:CCFX 中国金融期货交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param varUniCode: 品种编码,与getFutuVarirfCCXE关联,可获取期货品种基本信息,可以是列表,secID、ticker、exchangeCD、varUniCode至少选择一个
    :param contractStatus: 合约状态:L 上市；DE 已退市；UN 未上市。对应DataAPI.SysCodeGet.codeTypeID=10005。,可以是列表,可空
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
    requestString.append('/api/future/getFutuIrfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&varUniCode=")
    if hasattr(varUniCode,'__iter__') and not isinstance(varUniCode, str):
        if len(varUniCode) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = varUniCode
            requestString.append(None)
        else:
            requestString.append(','.join(varUniCode))
    else:
        requestString.append(varUniCode)
    requestString.append("&contractStatus=")
    if hasattr(contractStatus,'__iter__') and not isinstance(contractStatus, str):
        if len(contractStatus) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = contractStatus
            requestString.append(None)
        else:
            requestString.append(','.join(contractStatus))
    else:
        requestString.append(contractStatus)
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
            api_base.handle_error(csvString, 'FutuIrfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuIrfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secFullName', u'secFullNameEn', u'secShortName', u'secShortNameEn', u'cnSpell', u'ticker', u'exchangeCD', u'varUniCode', u'listDate', u'lastTradeDate', u'deliDate', u'contractStatus', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secFullName': 'str','secFullNameEn': 'str','secShortName': 'str','secShortNameEn': 'str','cnSpell': 'str','ticker': 'str','exchangeCD': 'str','varUniCode': 'str','listDate': 'str','lastTradeDate': 'str','deliDate': 'str','contractStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuVarcfCCXEGet(varUniCode = "", exchangeCD = "", deliMethod = "", field = "", pandas = "1"):
    """
    获取商品期货交易品种的基本要素信息，包括品种名称、品种代码、交易标的、报价单位、最小变动价位、涨跌停板幅度、合约乘数、最低交易保证金率、交割方式、交易手续费、最小交割单位等。
    
    :param varUniCode: 品种编码,与getFutuCfCCXE表'VAR_UNI_CODE'字段关联,可获取该品种期货合约信息,可以是列表,varUniCode、exchangeCD至少选择一个
    :param exchangeCD: 交易所,如:XDCE 大连商品交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,varUniCode、exchangeCD至少选择一个
    :param deliMethod: 交割方式:P 实物交割；C 现金结算。对应DataAPI.SysCodeGet.codeTypeID=60002。,可以是列表,可空
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
    requestString.append('/api/future/getFutuVarcfCCXE.csv?ispandas=1&') 
    requestString.append("varUniCode=")
    if hasattr(varUniCode,'__iter__') and not isinstance(varUniCode, str):
        if len(varUniCode) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = varUniCode
            requestString.append(None)
        else:
            requestString.append(','.join(varUniCode))
    else:
        requestString.append(varUniCode)
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&deliMethod=")
    if hasattr(deliMethod,'__iter__') and not isinstance(deliMethod, str):
        if len(deliMethod) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = deliMethod
            requestString.append(None)
        else:
            requestString.append(','.join(deliMethod))
    else:
        requestString.append(deliMethod)
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
            api_base.handle_error(csvString, 'FutuVarcfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuVarcfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'varUniCode', u'varName', u'varCode', u'exchangeCD', u'contMult', u'priceUnit', u'minChgPrice', u'chgPctLimit', u'contMonth', u'tradeTime', u'lastTdateDesc', u'lastTtimeDesc', u'deliDateDesc', u'deliMethod', u'deliAddr', u'deliGrade', u'minMarginRatio', u'tradeCommi', u'minDeliUnit', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'varUniCode': 'str','varName': 'str','varCode': 'str','exchangeCD': 'str','contMult': 'str','priceUnit': 'str','minChgPrice': 'str','chgPctLimit': 'str','contMonth': 'str','tradeTime': 'str','lastTdateDesc': 'str','lastTtimeDesc': 'str','deliDateDesc': 'str','deliMethod': 'str','deliAddr': 'str','deliGrade': 'str','minMarginRatio': 'str','tradeCommi': 'str','minDeliUnit': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuVarifCCXEGet(varUniCode = "", exchangeCD = "", deliMethod = "", field = "", pandas = "1"):
    """
    获取股指期货交易品种的基本要素信息，包括品种名称、品种代码、交易标的、报价单位、最小变动价位、涨跌停板幅度、合约乘数、最低交易保证金率、交割方式、交易手续费、最小交割单位等。
    
    :param varUniCode: 品种编码,与getFutuIfCCXE表'VAR_UNI_CODE'字段关联,可获取该品种期货合约信息,可以是列表,varUniCode、exchangeCD至少选择一个
    :param exchangeCD: 交易所,如:CCFX 中国金融期货交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,varUniCode、exchangeCD至少选择一个
    :param deliMethod: 交割方式:P 实物交割；C 现金结算。对应DataAPI.SysCodeGet.codeTypeID=60002。,可以是列表,可空
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
    requestString.append('/api/future/getFutuVarifCCXE.csv?ispandas=1&') 
    requestString.append("varUniCode=")
    if hasattr(varUniCode,'__iter__') and not isinstance(varUniCode, str):
        if len(varUniCode) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = varUniCode
            requestString.append(None)
        else:
            requestString.append(','.join(varUniCode))
    else:
        requestString.append(varUniCode)
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&deliMethod=")
    if hasattr(deliMethod,'__iter__') and not isinstance(deliMethod, str):
        if len(deliMethod) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = deliMethod
            requestString.append(None)
        else:
            requestString.append(','.join(deliMethod))
    else:
        requestString.append(deliMethod)
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
            api_base.handle_error(csvString, 'FutuVarifCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuVarifCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'varUniCode', u'varName', u'varCode', u'exchangeCD', u'contMult', u'priceUnit', u'minChgPrice', u'chgPctLimit', u'contMonth', u'tradeTime', u'lastTdateDesc', u'lastTtimeDesc', u'deliDateDesc', u'deliMethod', u'deliAddr', u'deliGrade', u'minMarginRatio', u'tradeCommi', u'minDeliUnit', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'varUniCode': 'str','varName': 'str','varCode': 'str','exchangeCD': 'str','contMult': 'str','priceUnit': 'str','minChgPrice': 'str','chgPctLimit': 'str','contMonth': 'str','tradeTime': 'str','lastTdateDesc': 'str','lastTtimeDesc': 'str','deliDateDesc': 'str','deliMethod': 'str','deliAddr': 'str','deliGrade': 'str','minMarginRatio': 'str','tradeCommi': 'str','minDeliUnit': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuVarirfCCXEGet(varUniCode = "", exchangeCD = "", deliMethod = "", field = "", pandas = "1"):
    """
    获取国债期货交易品种的基本要素信息，包括品种名称、品种代码、交易标的、报价单位、最小变动价位、涨跌停板幅度、合约乘数、最低交易保证金率、交割方式、交易手续费、最小交割单位等。
    
    :param varUniCode: 品种编码,与getFutuIrfCCXE表'VAR_UNI_CODE'字段关联,可获取该品种期货合约信息,可以是列表,varUniCode、exchangeCD至少选择一个
    :param exchangeCD: 交易所,如:CCFX 中国金融期货交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,varUniCode、exchangeCD至少选择一个
    :param deliMethod: 交割方式:P 实物交割；C 现金结算。对应DataAPI.SysCodeGet.codeTypeID=60002。,可以是列表,可空
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
    requestString.append('/api/future/getFutuVarirfCCXE.csv?ispandas=1&') 
    requestString.append("varUniCode=")
    if hasattr(varUniCode,'__iter__') and not isinstance(varUniCode, str):
        if len(varUniCode) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = varUniCode
            requestString.append(None)
        else:
            requestString.append(','.join(varUniCode))
    else:
        requestString.append(varUniCode)
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
    requestString.append("&deliMethod=")
    if hasattr(deliMethod,'__iter__') and not isinstance(deliMethod, str):
        if len(deliMethod) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = deliMethod
            requestString.append(None)
        else:
            requestString.append(','.join(deliMethod))
    else:
        requestString.append(deliMethod)
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
            api_base.handle_error(csvString, 'FutuVarirfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuVarirfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'varUniCode', u'varName', u'varCode', u'exchangeCD', u'contMult', u'priceUnit', u'minChgPrice', u'chgPctLimit', u'contMonth', u'tradeTime', u'lastTdateDesc', u'lastTtimeDesc', u'deliDateDesc', u'deliMethod', u'deliAddr', u'deliGrade', u'minMarginRatio', u'tradeCommi', u'minDeliUnit', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'varUniCode': 'str','varName': 'str','varCode': 'str','exchangeCD': 'str','contMult': 'str','priceUnit': 'str','minChgPrice': 'str','chgPctLimit': 'str','contMonth': 'str','tradeTime': 'str','lastTdateDesc': 'str','lastTtimeDesc': 'str','deliDateDesc': 'str','deliMethod': 'str','deliAddr': 'str','deliGrade': 'str','minMarginRatio': 'str','tradeCommi': 'str','minDeliUnit': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuChgCfCCXEGet(secID = "", ticker = "", secInfoType = "", isNew = "", field = "", pandas = "1"):
    """
    获取商品期货合约指标的历史变动数据，包括合约乘数、交易保证金、涨跌停板幅度、交易手续费、交割手续费等。
    
    :param secID: 合约展示代码,可通过合约代码和证券类型在DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker至少选择一个
    :param ticker: 合约代码,如:cu1509 沪铜1509合约,可以是列表,secID、ticker至少选择一个
    :param secInfoType: 证券指标,如:”0601“，表示 交易保证金。对应DataAPI.SysCodeGet.codeTypeID=10013。,可空
    :param isNew: 是否最新:1 是；0 否。对应DataAPI.SysCodeGet.codeTypeID=10007。,可以是列表,可空
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
    requestString.append('/api/future/getFutuChgCfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(secInfoType, str) and not isinstance(secInfoType, unicode):
        secInfoType = str(secInfoType)

    requestString.append("&secInfoType=%s"%(secInfoType))
    requestString.append("&isNew=")
    if hasattr(isNew,'__iter__') and not isinstance(isNew, str):
        if len(isNew) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isNew
            requestString.append(None)
        else:
            requestString.append(','.join(isNew))
    else:
        requestString.append(isNew)
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
            api_base.handle_error(csvString, 'FutuChgCfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuChgCfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secShortName', u'secShortNameEn', u'ticker', u'exchangeCD', u'changeDate', u'secInfoType', u'value', u'valueUnit', u'isNew', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','exchangeCD': 'str','changeDate': 'str','secInfoType': 'str','valueUnit': 'str','isNew': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuChgIfCCXEGet(secID = "", ticker = "", secInfoType = "", isNew = "", field = "", pandas = "1"):
    """
    获取股指期货合约指标的历史变动数据，包括合约乘数、交易保证金、涨跌停板幅度、交易手续费、交割手续费等。
    
    :param secID: 合约展示代码,可通过合约代码和证券类型在DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker至少选择一个
    :param ticker: 合约代码,如:IF1412 沪深300股指期货1412合约,可以是列表,secID、ticker至少选择一个
    :param secInfoType: 证券指标,如:”0601",表示合约保证金。对应DataAPI.SysCodeGet.codeTypeID=10013。,可空
    :param isNew: 是否最新:1 是；0 否。对应DataAPI.SysCodeGet.codeTypeID=10007。,可以是列表,可空
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
    requestString.append('/api/future/getFutuChgIfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(secInfoType, str) and not isinstance(secInfoType, unicode):
        secInfoType = str(secInfoType)

    requestString.append("&secInfoType=%s"%(secInfoType))
    requestString.append("&isNew=")
    if hasattr(isNew,'__iter__') and not isinstance(isNew, str):
        if len(isNew) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isNew
            requestString.append(None)
        else:
            requestString.append(','.join(isNew))
    else:
        requestString.append(isNew)
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
            api_base.handle_error(csvString, 'FutuChgIfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuChgIfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secShortName', u'secShortNameEn', u'ticker', u'exchangeCD', u'changeDate', u'secInfoType', u'value', u'valueUnit', u'isNew', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','exchangeCD': 'str','changeDate': 'str','secInfoType': 'str','valueUnit': 'str','isNew': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FutuChgIrfCCXEGet(secID = "", ticker = "", secInfoType = "", isNew = "", field = "", pandas = "1"):
    """
    获取国债期货合约指标的历史变动数据，包括合约乘数、交易保证金、涨跌停板幅度、交易手续费、交割手续费等。
    
    :param secID: 合约展示代码,可通过合约代码和证券类型在DataAPI.CCXE.SecIDCCXEGet中获取,可以是列表,secID、ticker至少选择一个
    :param ticker: 合约代码,如:TF1412 票面利率3%的5年期名义中期国债期货1412合约,可以是列表,secID、ticker至少选择一个
    :param secInfoType: 证券指标,如:”0601",表示合约保证金。对应DataAPI.SysCodeGet.codeTypeID=10013。,可空
    :param isNew: 是否最新:1 是；0 否。对应DataAPI.SysCodeGet.codeTypeID=10007。,可以是列表,可空
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
    requestString.append('/api/future/getFutuChgIrfCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    if not isinstance(secInfoType, str) and not isinstance(secInfoType, unicode):
        secInfoType = str(secInfoType)

    requestString.append("&secInfoType=%s"%(secInfoType))
    requestString.append("&isNew=")
    if hasattr(isNew,'__iter__') and not isinstance(isNew, str):
        if len(isNew) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = isNew
            requestString.append(None)
        else:
            requestString.append(','.join(isNew))
    else:
        requestString.append(isNew)
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
            api_base.handle_error(csvString, 'FutuChgIrfCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FutuChgIrfCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secShortName', u'secShortNameEn', u'ticker', u'exchangeCD', u'changeDate', u'secInfoType', u'value', u'valueUnit', u'isNew', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','exchangeCD': 'str','secInfoType': 'str','valueUnit': 'str','isNew': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FstTotalCCXEGet(tradeDate = "", exchangeCD = "", field = "", pandas = "1"):
    """
    获取上海、深圳交易所公布的每个交易日的融资融券交易汇总的信息，包括成交量、成交金额。本交易日可获取前一交易日的数据。
    
    :param tradeDate: 交易日期，默认为前1天，输入格式“YYYYMMDD”,可空
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,可空
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
    requestString.append('/api/margin/getFstTotalCCXE.csv?ispandas=1&') 
    if not isinstance(tradeDate, str) and not isinstance(tradeDate, unicode):
        tradeDate = str(tradeDate)

    requestString.append("tradeDate=%s"%(tradeDate))
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
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
            api_base.handle_error(csvString, 'FstTotalCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FstTotalCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'tradeDate', u'exchangeCD', u'currencyCD', u'finVal', u'finBuyVal', u'secVol', u'secVal', u'secSellVol', u'tradeVal']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','exchangeCD': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FstDetailCCXEGet(secID = "", ticker = "", assetClass = "", exchangeCD = "", tradeDate = "", field = "", pandas = "1"):
    """
    获取上海、深圳交易所公布的每个交易日的融资融券交易具体的信息，包括标的证券信息、融资融券金额以及数量方面的数据。本交易日可获取前一交易日的数据。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,secID、ticker、assetClass、exchangeCD至少选择一个
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,secID、ticker、assetClass、exchangeCD至少选择一个
    :param assetClass: 证券类型。例如，E-股票,B-债券,F-基金。对应DataAPI.SysCodeGet.codeTypeID=10001。,可以是列表,secID、ticker、assetClass、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、assetClass、exchangeCD至少选择一个
    :param tradeDate: 交易日期，默认为前1天，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/margin/getFstDetailCCXE.csv?ispandas=1&') 
    requestString.append("secID=")
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
    requestString.append("&assetClass=")
    if hasattr(assetClass,'__iter__') and not isinstance(assetClass, str):
        if len(assetClass) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = assetClass
            requestString.append(None)
        else:
            requestString.append(','.join(assetClass))
    else:
        requestString.append(assetClass)
    requestString.append("&exchangeCD=")
    if hasattr(exchangeCD,'__iter__') and not isinstance(exchangeCD, str):
        if len(exchangeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = exchangeCD
            requestString.append(None)
        else:
            requestString.append(','.join(exchangeCD))
    else:
        requestString.append(exchangeCD)
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
            api_base.handle_error(csvString, 'FstDetailCCXEGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FstDetailCCXEGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'tradeDate', u'secID', u'ticker', u'assetClass', u'exchangeCD', u'secShortName', u'secShortNameEn', u'currencyCD', u'finVal', u'finBuyVal', u'finRefundVal', u'secVol', u'secSellVol', u'secRefundVol', u'secVal', u'tradeVal']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tradeDate': 'str','secID': 'str','ticker': 'str','assetClass': 'str','exchangeCD': 'str','secShortName': 'str','secShortNameEn': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

