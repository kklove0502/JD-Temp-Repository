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

__doc__="巨灵财经"
def FundJLGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取基金的基本档案信息，包含基金名称、交易代码、所属类别、设立信息、上市信息、相关机构、投资描述等信息。数据更新频率为不定期。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003",可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fund/getFundJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'ticker1', u'ticker2', u'ticker3', u'ticker4', u'ticker5', u'subscriptionCodeFront', u'subscriptionCodeBack', u'secFullName', u'secShortName', u'cnSpell', u'secShortName_2', u'cnSpell2', u'exchangeCD', u'listDate', u'listStatus', u'managementCompanyID', u'managementCompany', u'custodianID', u'custodian', u'initEstablishDate', u'establishDate', u'duration', u'expireDate', u'fundID', u'type', u'isQdii', u'investType', u'investStyle', u'currencyCD', u'isInnovativeFund', u'isUmbrellaFund', u'isTransFund', u'profitDistributionCD', u'profitConvertCD', u'fundProfile', u'ISIN']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','ticker1': 'str','ticker2': 'str','ticker3': 'str','ticker4': 'str','ticker5': 'str','subscriptionCodeFront': 'str','subscriptionCodeBack': 'str','secFullName': 'str','secShortName': 'str','cnSpell': 'str','secShortName_2': 'str','cnSpell2': 'str','exchangeCD': 'str','listStatus': 'str','managementCompany': 'str','custodian': 'str','type': 'str','investType': 'str','investStyle': 'str','currencyCD': 'str','profitDistributionCD': 'str','profitConvertCD': 'str','fundProfile': 'str','ISIN': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundNavJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取某只基金的历史净值信息（货币型和理财型基金除外），包括单位净值、累计净值和基金资产净值等。收录了1998年以来的历史数据，数据更新频率为日。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundNavJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundNavJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundNavJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'endDate', u'nav', u'accumNav', u'adjustNav', u'netAsset']
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

def FundDivmJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取某只货币型或理财型基金历史收益信息，包括每万份基金单位收益、最近7日年化收益率和基金资产净值。收录了2004年以来的历史数据，数据更新频率为日。不输入日期则默认获取近一年以来的历史数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 输入一个或多个基金代码，用","分隔，如"000001"、"000001,000003",可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fund/getFundDivmJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundDivmJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundDivmJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'endDate', u'dailyProfit', u'weeklyYield', u'netAsset', u'classNameCD', u'classNameDesc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','classNameCD': 'str','classNameDesc': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundDividendJLGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取基金的历史分红信息(含预案公告和实施公告),包括截止日期,派息对象,基金份额基数,每10份基金份额派发红利,分红方式,股权登记日等。收录了1994年以来的历史数据，数据更新频率为不定期。
    
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
    requestString.append('/api/fund/getFundDividendJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundDividendJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundDividendJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'serialNumber', u'publishDate', u'distBaseDate', u'dividendYear', u'beneficiary', u'divTypeCD', u'dividendBfTax', u'dividendAfTax', u'navAfDiv', u'regDate', u'exDivDate', u'exDivDateField', u'paymentDateOTC', u'paymentDateField', u'paymentDesc', u'reinvestDate', u'reinvRedemDate', u'reinvAcctDate', u'convertDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','beneficiary': 'str','divTypeCD': 'str','paymentDesc': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundSplitJLGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    获取基金拆分折算的相关历史信息，包括拆分折算日、拆分折算比例、拆分折算方案进度等内容。收录了2005年以来的历史数据，数据更新频率为不定期。
    
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
    requestString.append('/api/fund/getFundSplitJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundSplitJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundSplitJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'infoSource', u'splitTypeCD', u'splitDate', u'fundSplited', u'splitRatio', u'resultOutcomeDate', u'chgRegDate', u'netAsset', u'sharesBf', u'sharesAf', u'navBf', u'navAf', u'accumNavBf', u'accumNavAf', u'accumNavAfDesc', u'splitDesc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','infoSource': 'str','splitTypeCD': 'str','fundSplited': 'str','accumNavAfDesc': 'str','splitDesc': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundAssetsJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金的资产配置信息,包括持有股票,债券,基金,权证和银行存款及清算备付金的市值及其占资产总资产和基金净值的比例等信息。收录了1998年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundAssetsJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundAssetsJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundAssetsJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'custodianCheckDate', u'reportDate', u'repTypeCD', u'netAsset', u'totalAsset', u'totalShare', u'equityInvestMarketValue', u'equityInvestRatioInTA', u'stockMarketValue', u'stockRatioInTA', u'stockRatioInNA', u'fixIncomeMarketValue', u'fixIncomeRatioInTA', u'bondMarketValue', u'bondRatioInTA', u'bondRatioInNA', u'cashMarketValue', u'cashRatioInTA', u'cashRatioInNA', u'otherMarketValue', u'otherRatioInTA', u'otherRatioInNA', u'resaleSecMarketValue', u'resaleSecRatioInTA', u'resaleSecRatioInNA', u'warrantMarketValue', u'warrantRatioInTA', u'warrantRatioInNA', u'holdFundMarketValue', u'holdFundRatioInTA', u'holdFundRatioInNA', u'absMarketValue', u'absRatioInTA', u'absRatioInNA', u'moneyMarketInstument', u'moneyMarketInstuRatioInTA', u'payableReceivable', u'payableReceivableRatioInNA', u'convtBondMarketValue', u'convtBondRatioInNA', u'govBondMarketValue', u'govBondRatioInNA', u'drMarketValue', u'drRatioInTA', u'forwardsMarketValue', u'forwardsRatioInTA', u'futuresMarketValue', u'fururesRatioInTA', u'optionMarketValue', u'optionRatioInTA', u'derivativesMarketValue', u'derivativesRatioInTA', u'otherInvMarketValue', u'otherInvRatioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','repTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsEJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金投资组合的股票配置明细，包括配置股票内部代码、持股数量、市值等内容。收录了1998年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundHoldingsEJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundHoldingsEJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsEJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'repTypeCD', u'equitySecID', u'equityTicker', u'equityExchangeCD', u'equityShortName', u'holdVolume', u'marketValue', u'ratioInNA', u'volumeRatioInAShares']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','repTypeCD': 'str','equitySecID': 'str','equityTicker': 'str','equityExchangeCD': 'str','equityShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsEIndexJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金投资组合中指数投资那部分的股票配置明细，包括配置股票代码、持股数量、市值等内容。收录了1998年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundHoldingsEIndexJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundHoldingsEIndexJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsEIndexJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'repTypeCD', u'equitySecID', u'equityTicker', u'equityExchangeCD', u'equityShortName', u'holdVolume', u'marketValue', u'ratioInNA', u'volumeRatioInAShares']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','repTypeCD': 'str','equitySecID': 'str','equityTicker': 'str','equityExchangeCD': 'str','equityShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsEActiveJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金投资组合中积极投资那部分的股票配置明细，包括配置股票代码、持股数量、市值等内容。收录了1999年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundHoldingsEActiveJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundHoldingsEActiveJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsEActiveJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'repTypeCD', u'equitySecID', u'equityTicker', u'equityExchangeCD', u'equityShortName', u'holdVolume', u'marketValue', u'ratioInNA', u'volumeRatioInAShares']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','repTypeCD': 'str','equitySecID': 'str','equityTicker': 'str','equityExchangeCD': 'str','equityShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsBJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金投资组合中的债券配置信息，包括债券代码、债券数量、市值、摊余成本及其占资产净值比等。收录了2002年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundHoldingsBJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundHoldingsBJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsBJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'repTypeCD', u'serialNumber', u'bondSecID', u'bondTicker', u'bondExchangeCD', u'bondShortName', u'isConvertible', u'holdVolume', u'marketValue', u'ratioInNA', u'paymentTerm', u'amortizedCost', u'amortizedCostRatioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','repTypeCD': 'str','bondSecID': 'str','bondTicker': 'str','bondExchangeCD': 'str','bondShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FundHoldingsFJLGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取基金投资组合中的基金配置信息。收录了2007年以来的历史数据，数据更新频率为季度。不输入日期则默认获取近一年以来的历史数据。
    
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
    requestString.append('/api/fund/getFundHoldingsFJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'FundHoldingsFJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FundHoldingsFJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'publishDate', u'reportDate', u'repTypeCD', u'holdFundSecID', u'holdFundTicker', u'holdFundExchangeCD', u'holdFundShortName', u'holdVolume', u'marketValue', u'ratioInNA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','repTypeCD': 'str','holdFundSecID': 'str','holdFundTicker': 'str','holdFundExchangeCD': 'str','holdFundShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtBSJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间资产负债表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、本表为资产负债表通用表，包括金融类上市公司部分通用的科目内容； 3、收集合并报表、母公司报表对应的数据； 4、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 5、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期,起始时间,如‘20121231’,可空
    :param endDate: 会计期间截止日期,结束时间,如‘20131231’,可空
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
    requestString.append('/api/listedCorp/getFdmtBSJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
            api_base.handle_error(csvString, 'FdmtBSJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtBSJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'cashCEquiv', u'tradingFA', u'notesReceiv', u'divReceiv', u'intReceiv', u'AR', u'othReceiv', u'prepayment', u'inventories', u'BBA', u'deferExp', u'NCAWithin1Y', u'othCA', u'CAE', u'TCA', u'availForSaleFA', u'htmInvest', u'investRealEstate', u'LTReceiv', u'LTEquityInvest', u'fixedAssets', u'constMaterials', u'CIP', u'fixedAssetsDisp', u'producBiolAssets', u'oilAndGasAssets', u'intanAssets', u'RD', u'goodwill', u'LTAmorExp', u'deferTaxAssets', u'othNCA', u'NCAE', u'TNCA', u'TAssets', u'STBorr', u'tradingFL', u'notesPayable', u'AP', u'STBondPayable', u'advanceReceipts', u'payrollPayable', u'divPayable', u'taxesPayable', u'intPayable', u'othPayable', u'accrExp', u'estimatedLiabST', u'NCLWithin1Y', u'deferIncomeST', u'othCl', u'CLE', u'TCL', u'LTBorr', u'bondPayable', u'LTPayable', u'specificPayables', u'deferTaxLiab', u'estimatedLiab', u'deferIncome', u'othNCL', u'NCLE', u'TNCL', u'TLiab', u'paidInCapital', u'capitalReser', u'surplusReser', u'treasuryShare', u'specialReser', u'unInvestlLoss', u'retainedEarnings', u'forexDiffer', u'othEffectSe', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TLiabEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','endDateRep': 'str','endDate': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtBSBankJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了银行业上市公司定期报告中各个期间资产负债表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000001.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000001',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtBSBankJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtBSBankJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtBSBankJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'BsecShortName', u'tickerB', u'AsecShortName', u'infoSource', u'endDateRep', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'cashCEquiv', u'preciMetals', u'loanToOthBankFI', u'tradingFA', u'refundDepos', u'derivAssets', u'purResaleFA', u'deposInOthBFI', u'notesReceiv', u'othReceiv', u'AR', u'intReceiv', u'divReceiv', u'prepayment', u'inventories', u'deferExp', u'LTEquityInvest', u'disburLA', u'availForSaleFA', u'htmInvest', u'LTReceiv', u'investRealEstate', u'fixedAssets', u'CIP', u'constMaterials', u'fixedAssetsDisp', u'intanAssets', u'transacSeatFee', u'RD', u'goodwill', u'LTAmorExp', u'refundCapDepos', u'deferTaxAssets', u'investAsReceiv', u'othAssets', u'TAssets', u'deposFrOthBFI', u'STBorr', u'pledgeBorr', u'loanFrOthBankFI', u'tradingFL', u'derivLiab', u'soldForRepurFA', u'CBBorr', u'depos', u'intPayable', u'divPayable', u'fundsSecTradAgen', u'fundsSecUndwAgen', u'notesPayable', u'AP', u'advanceReceipts', u'payrollPayable', u'taxesPayable', u'othPayable', u'accrExp', u'estimatedLiab', u'deposReceiv', u'LTBorr', u'LTPayable', u'bondPayable', u'deferTaxLiab', u'othLiab', u'TLiab', u'paidInCapital', u'capitalReser', u'treasuryShare', u'ordinRiskReser', u'retainedEarnings', u'forexDiffer', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TLiabEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','BsecShortName': 'str','tickerB': 'str','AsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtBSNbankJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了非银行业金融类（保险、证券、信托等）上市公司定期报告中各个期间资产负债表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'600369.XSHG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'600369',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtBSNbankJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtBSNbankJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtBSNbankJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'cashCEquiv', u'clientDepos', u'PHPledgeLoans', u'settProv', u'preciMetals', u'loanToOthBankFI', u'tradingFA', u'refundDepos', u'derivAssets', u'purResaleFA', u'premiumReceiv', u'subrogRecoReceiv', u'reinsurReceiv', u'reinsurReserReceiv', u'RRReinsUnePrem', u'RRReinsOutstdCla', u'RRReinsLinsLiab', u'RRReinsLthinsLiab', u'notesReceiv', u'othReceiv', u'AR', u'intReceiv', u'divReceiv', u'prepayment', u'inventories', u'deferExp', u'investAsReceiv', u'NCAWithin1Y', u'othCA', u'TCA', u'disburLA', u'availForSaleFA', u'htmInvest', u'investRealEstate', u'LTEquityInvest', u'fixedAssets', u'CIP', u'fixedAssetsDisp', u'intanAssets', u'transacSeatFee', u'RD', u'goodwill', u'LTAmorExp', u'refundCapDepos', u'deferTaxAssets', u'fixedTermDepos', u'othNCA', u'TNCA', u'othAssets', u'TAssets', u'deposFrOthBFI', u'STBorr', u'pledgeBorr', u'loanFrOthBankFI', u'tradingFL', u'derivLiab', u'soldForRepurFA', u'CBBorr', u'depos', u'intPayable', u'divPayable', u'fundsSecTradAgen', u'fundsSecUndwAgen', u'notesPayable', u'AP', u'advanceReceipts', u'premReceivAdva', u'commisPayable', u'reinsurPayable', u'insurReser', u'payrollPayable', u'taxesPayable', u'othPayable', u'accrExp', u'estimatedLiabST', u'indemAccPayable', u'policyDivPayable', u'reserUnePrem', u'reserOutstdClaims', u'deferIncomeST', u'NCLWithin1Y', u'othCl', u'TCL', u'reserLinsLiab', u'reserLthinsLiab', u'PHInvest', u'deposReceiv', u'LTBorr', u'LTPayable', u'bondPayable', u'deferTaxLiab', u'deferIncome', u'estimatedLiab', u'othNCL', u'TNCL', u'othLiab', u'TLiab', u'paidInCapital', u'capitalReser', u'treasuryShare', u'surplusReser', u'ordinRiskReser', u'retainedEarnings', u'forexDiffer', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TLiabEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtISJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间利润表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、本表为利润通用表，包括金融类上市公司部分通用的科目内容； 3、收集合并报表、母公司报表对应的数据； 4、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 5、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtISJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtISJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtISJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'TRevenue', u'revenue', u'specOR', u'TCOGS', u'COGS', u'specOC', u'bizTaxSurchg', u'sellExp', u'adminExp', u'finanExp', u'assetsImpairLoss', u'FValueChgGain', u'investIncome', u'AJInvestIncome', u'forexGain', u'othEffectOP', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'NCADisploss', u'othEffectTP', u'TProfit', u'incomeTax', u'unInvestLoss', u'othEffectNPP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'basicEPS', u'dilutedEPS', u'othComprIncome', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtISBankJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了银行业上市公司定期报告中各个期间利润表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000001.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000001',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtISBankJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtISBankJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtISBankJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'revenue', u'NIntIncome', u'intIncome', u'intExp', u'NCommisIncome', u'commisIncome', u'commisExp', u'NTrustIncome', u'investIncome', u'AJInvestIncome', u'FValueChgGain', u'forexGain', u'othRevenue', u'payout', u'genlAdminExp', u'finanExp', u'assetsImpairLoss', u'othCost', u'bizTaxSurchg', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'NCADisploss', u'TProfit', u'incomeTax', u'othEffectNP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'basicEPS', u'dilutedEPS', u'othComprIncome', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtISNbankJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了非银行业金融类（保险、证券、信托等）上市公司定期报告中各个期间利润表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'600369.XSHG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'600369',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtISNbankJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtISNbankJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtISNbankJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'revenue', u'NIntIncome', u'intIncome', u'intExp', u'premEarned', u'grossPremWrit', u'reinsIncome', u'reinsur', u'unePremReser', u'NCommisIncome', u'commisIncome', u'commisExp', u'NSecTaIncome', u'NUndwrtSecIncome', u'NTrustIncome', u'investIncome', u'AJInvestIncome', u'FValueChgGain', u'forexGain', u'realEstateIncome', u'othRevenue', u'payout', u'insurCommisExp', u'compensPayout', u'compensPayoutRefu', u'reserInsurLiab', u'insurLiabReserRefu', u'reinsurExp', u'premRefund', u'policyDivPayt', u'genlAdminExp', u'reinsCostRefund', u'sellExp', u'finanExp', u'assetsImpairLoss', u'othCost', u'bizTaxSurchg', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'othEffectTP', u'TProfit', u'incomeTax', u'othEffectNP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'basicEPS', u'dilutedEPS', u'othComprIncome', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtCFJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了上市公司定期报告中各个期间现金流量表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、本表为利润通用表，包括金融类上市公司部分通用的科目内容； 3、收集合并报表、母公司报表对应的数据； 4、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 5、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtCFJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtCFJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtCFJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'CFrSaleGS', u'refundOfTax', u'CFrOthOperateA', u'CInfFrOperateA', u'CPaidGS', u'CPaidToForEmpl', u'CPaidForTaxes', u'CPaidForOthOpA', u'COutfOperateA', u'NCFOperateA', u'procSellInvest', u'gainInvest', u'dispFixAssetsOth', u'NDispSubsOthBizC', u'CFrOthInvestA', u'CInfFrInvestA', u'purFixAssetsOth', u'CPaidInvest', u'NCPaidAcquis', u'CPaidOthInvestA', u'COutfFrInvestA', u'NCFFrInvestA', u'CFrCapContr', u'CFrBorr', u'CFrIssueBond', u'CFrOthFinanA', u'CInfFrFinanA', u'CPaidForDebts', u'CPaidDivProfInt', u'CPaidOthFinanA', u'COutfFrFinanA', u'NCFFrFinanA', u'forexEffects', u'othEffectCE', u'NChangeInCash', u'NCEBegBal', u'NCEEndBal', u'convDebtCapi', u'convBonds1Y', u'finanLeaFA', u'NIncome', u'minorityGain', u'FAOilBiolDepr', u'FADepr', u'intanAssetsAmor', u'LTAmorExpAmor', u'amorExpDecr', u'accrExpIncr', u'dispFAOthLoss', u'FAWritOff', u'FValueChgLoss', u'finanExp', u'invLoss', u'deferTADecr', u'deferTLIncr', u'invenDecr', u'operReceiDecr', u'operPayaIncr', u'other', u'NCFOperateANotes', u'CEndBal', u'CBegBal', u'CEEndBal', u'CEBegBal', u'specC', u'NChangeInCashNotes']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtCFBankJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了银行业上市公司定期报告中各个期间现金流量表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000001.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000001',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtCFBankJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtCFBankJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtCFBankJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'NDeposIncrCFI', u'NIncrBorrFrCB', u'NIncBorrOthFI', u'IFCCashIncr', u'refundOfTax', u'drawBackLoansC', u'NIncDispTradFA', u'NIncDispFAFS', u'CFrOthOperateA', u'CInfFrOperateA', u'NIncDisburOfLA', u'CPaidForTaxes', u'netIncrDeposInFI', u'CPaidIFC', u'CPaidGS', u'CPaidToForEmpl', u'CPaidForOthOpA', u'COutfOperateA', u'NCFOperateA', u'procSellInvest', u'gainInvest', u'dispFixAssetsOth', u'NDispSubsOthBizC', u'CFrOthInvestA', u'CInfFrInvestA', u'CPaidInvest', u'purFixAssetsOth', u'NCPaidAcquis', u'CPaidOthInvestA', u'COutfFrInvestA', u'NCFFrInvestA', u'CFrCapContr', u'CFrMinoSSubs', u'CFrIssueBond', u'CFrBorr', u'CFrOthFinanA', u'CInfFrFinanA', u'CPaidForDebts', u'CPaidDivProfInt', u'divProfSubsMinoS', u'CPaidOthFinanA', u'COutfFrFinanA', u'NCFFrFinanA', u'forexEffects', u'NChangeInCash', u'NCEBegBal', u'NCEEndBal', u'NIncome', u'minorityGain', u'FADepr', u'FAOilBiolDepr', u'intanAssetsAmor', u'LTAmorExpAmor', u'amorExpDecr', u'accrExpIncr', u'dispFAOthLoss', u'FAWritOff', u'FValueChgLoss', u'finanExp', u'invLoss', u'invenDecr', u'operReceiDecr', u'operPayaIncr', u'deferTADecr', u'deferTLIncr', u'other', u'NCFOperateANotes', u'CEndBal', u'CBegBal', u'CEEndBal', u'CEBegBal', u'NCEEndBalNotes', u'NCEBegBalNotes', u'NChangeInCashNotes']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtCFNbankJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了非银行业金融类（保险、证券、信托等）上市公司定期报告中各个期间现金流量表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、收集合并报表、母公司报表对应的数据； 3、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 4、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'600369.XSHG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'600369',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getFdmtCFNbankJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'FdmtCFNbankJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtCFNbankJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'NIncBorrOthFI', u'premFrOrigContr', u'NReinsurPrem', u'NIncPhDeposInv', u'NIncDispTradFA', u'IFCCashIncr', u'NCapIncrRepur', u'CFrSaleGS', u'refundOfTax', u'drawBackLoansC', u'CFrOthOperateA', u'CInfFrOperateA', u'NIncDisburOfLA', u'netIncrDeposInFI', u'origContrCIndem', u'CPaidPolDiv', u'CPaidIFC', u'CPaidForTaxes', u'CPaidGS', u'CPaidToForEmpl', u'CPaidForOthOpA', u'COutfOperateA', u'NCFOperateA', u'procSellInvest', u'gainInvest', u'dispFixAssetsOth', u'NDispSubsOthBizC', u'CFrOthInvestA', u'CInfFrInvestA', u'CPaidInvest', u'NIncrPledgeLoan', u'purFixAssetsOth', u'NCPaidAcquis', u'CPaidOthInvestA', u'COutfFrInvestA', u'NCFFrInvestA', u'CFrCapContr', u'CFrMinoSSubs', u'CFrIssueBond', u'CFrBorr', u'CFrOthFinanA', u'CInfFrFinanA', u'CPaidForDebts', u'CPaidDivProfInt', u'divProfSubsMinoS', u'CPaidOthFinanA', u'COutfFrFinanA', u'NCFFrFinanA', u'forexEffects', u'NChangeInCash', u'NCEBegBal', u'NCEEndBal', u'NIncome', u'minorityGain', u'FADepr', u'FAOilBiolDepr', u'intanAssetsAmor', u'LTAmorExpAmor', u'amorExpDecr', u'accrExpIncr', u'dispFAOthLoss', u'FAWritOff', u'FValueChgLoss', u'finanExp', u'invLoss', u'invenDecr', u'operReceiDecr', u'operPayaIncr', u'deferTADecr', u'deferTLIncr', u'other', u'NCFOperateANotes', u'CEndBal', u'CBegBal', u'CEEndBal', u'CEBegBal', u'NCEEndBalNotes', u'NCEBegBalNotes', u'NChangeInCashNotes']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondFdmtBSJLGet(secID = "", ticker = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了债券发行人定期报告中各个期间资产负债表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、本表为资产负债表通用表，包括金融类上市公司部分通用的科目内容； 3、收集合并报表、母公司报表对应的数据； 4、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 5、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'081601.XIBE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'081601',可以是列表,secID、ticker至少选择一个
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getBondFdmtBSJL.csv?ispandas=1&') 
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'BondFdmtBSJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondFdmtBSJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'infoSource', u'endDateRep', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'cashCEquiv', u'tradingFA', u'notesReceiv', u'divReceiv', u'intReceiv', u'AR', u'othReceiv', u'prepayment', u'inventories', u'BBA', u'deferExp', u'NCAWithin1Y', u'othCA', u'CAE', u'TCA', u'availForSaleFA', u'htmInvest', u'investRealEstate', u'LTReceiv', u'LTEquityInvest', u'fixedAssets', u'constMaterials', u'CIP', u'fixedAssetsDisp', u'producBiolAssets', u'oilAndGasAssets', u'intanAssets', u'RD', u'goodwill', u'LTAmorExp', u'deferTaxAssets', u'othNCA', u'NCAE', u'TNCA', u'TAssets', u'STBorr', u'tradingFL', u'notesPayable', u'AP', u'advanceReceipts', u'payrollPayable', u'divPayable', u'taxesPayable', u'intPayable', u'othPayable', u'STBondPayable', u'accrExp', u'estimatedLiabST', u'NCLWithin1Y', u'deferIncomeST', u'othCl', u'CLE', u'TCL', u'LTBorr', u'bondPayable', u'LTPayable', u'specificPayables', u'deferTaxLiab', u'estimatedLiab', u'deferIncome', u'othNCL', u'NCLE', u'TNCL', u'TLiab', u'paidInCapital', u'capitalReser', u'surplusReser', u'treasuryShare', u'specialReser', u'unInvestlLoss', u'retainedEarnings', u'forexDiffer', u'othEffectSe', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TLiabEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondFdmtISJLGet(secID = "", ticker = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了债券发行人定期报告中各个期间利润表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、本表为利润表通用表，包括金融类上市公司部分通用的科目内容； 3、收集合并报表、母公司报表对应的数据； 4、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 5、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'081601.XIBE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'081601',可以是列表,secID、ticker至少选择一个
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getBondFdmtISJL.csv?ispandas=1&') 
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'BondFdmtISJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondFdmtISJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'TRevenue', u'revenue', u'specOR', u'TCOGS', u'COGS', u'specOC', u'bizTaxSurchg', u'sellExp', u'adminExp', u'finanExp', u'assetsImpairLoss', u'FValueChgGain', u'investIncome', u'AJInvestIncome', u'forexGain', u'othEffectOP', u'operateProfit', u'NoperateIncome', u'NoperateExp', u'NCADisploss', u'othEffectTP', u'TProfit', u'incomeTax', u'unInvestLoss', u'othEffectNPP', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'basicEPS', u'dilutedEPS', u'othComprIncome', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondFdmtCFJLGet(secID = "", ticker = "", beginDate = "", endDate = "", beginDateRep = "", endDateRep = "", field = "", pandas = "1"):
    """
    1、依据2007年新会计准则收集了债券发行人定期报告中各个期间现金流量表数据，并依据新旧会计准则的科目对应关系，收录主要科目的历史对应数据； 2、本表为现金流量表通用表，包括金融类上市公司部分通用的科目内容； 3、收集合并报表、母公司报表对应的数据； 4、对于同一报告期内的数据只对最新报告中的本期数据做记录,例如:某公司2008年2月发布招股说明书,里面有2006年年度财务报告,若本公司在3月份的年报中又披漏2006年度财务数据,则覆盖招股说明书中的2006年度数据。 5、本表中单位为人民币元。
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'081601.XIBE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'081601',可以是列表,secID、ticker至少选择一个
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param beginDateRep: 报告的会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDateRep: 报告的会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/listedCorp/getBondFdmtCFJL.csv?ispandas=1&') 
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
    try:
        beginDateRep = beginDateRep.strftime('%Y%m%d')
    except:
        beginDateRep = beginDateRep.replace('-', '')
    requestString.append("&beginDateRep=%s"%(beginDateRep))
    try:
        endDateRep = endDateRep.strftime('%Y%m%d')
    except:
        endDateRep = endDateRep.replace('-', '')
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
            api_base.handle_error(csvString, 'BondFdmtCFJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondFdmtCFJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'infoSource', u'endDateRep', u'startDate', u'endDate', u'mergedFlag', u'parCom', u'currencyCD', u'CFrSaleGS', u'refundOfTax', u'CFrOthOperateA', u'CInfFrOperateA', u'CPaidGS', u'CPaidToForEmpl', u'CPaidForTaxes', u'CPaidForOthOpA', u'COutfOperateA', u'NCFOperateA', u'procSellInvest', u'gainInvest', u'dispFixAssetsOth', u'NDispSubsOthBizC', u'CFrOthInvestA', u'CInfFrInvestA', u'purFixAssetsOth', u'CPaidInvest', u'NCPaidAcquis', u'CPaidOthInvestA', u'COutfFrInvestA', u'NCFFrInvestA', u'CFrCapContr', u'CFrBorr', u'CFrIssueBond', u'CFrOthFinanA', u'CInfFrFinanA', u'CPaidForDebts', u'CPaidDivProfInt', u'CPaidOthFinanA', u'COutfFrFinanA', u'NCFFrFinanA', u'forexEffects', u'othEffectCE', u'NChangeInCash', u'NCEBegBal', u'NCEEndBal', u'convDebtCapi', u'convBonds1Y', u'finanLeaFA', u'NIncome', u'minorityGain', u'FAOilBiolDepr', u'FADepr', u'intanAssetsAmor', u'LTAmorExpAmor', u'amorExpDecr', u'accrExpIncr', u'dispFAOthLoss', u'FAWritOff', u'FValueChgLoss', u'finanExp', u'invLoss', u'deferTADecr', u'deferTLIncr', u'invenDecr', u'operReceiDecr', u'operPayaIncr', u'other', u'NCFOperateANotes', u'CEndBal', u'CBegBal', u'CEEndBal', u'CEBegBal', u'specC', u'NChangeInCashNotes']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','infoSource': 'str','mergedFlag': 'str','parCom': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtEEJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", publishDateBegin = "", publishDateEnd = "", field = "", pandas = "1"):
    """
    获取上市公司披露的业绩快报中的主要财务指标等其他数据，包括本期，去年同期，及本期与期初数值同比数据。（若上市公司同时发行债券等其他证券，也可通过其他证券代码查询快报数据）
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'600000',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param publishDateBegin: 证券交易所披露的信息发布日期,起始时间,如‘20130812’,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期,结束时间,默认为当前日期,如‘20140812’,可空
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
    requestString.append('/api/listedCorp/getFdmtEEJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'FdmtEEJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtEEJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'publishDate', u'endDate', u'itemCD', u'item', u'chng', u'chngCause', u'oprFinExp', u'serial_Num', u'revenue', u'primeOperRev', u'grossProfit', u'operateProfit', u'TProfit', u'NINCOMEAttrP', u'NINCOMECut', u'basicEPS', u'EPSDilu', u'EPSW', u'EPSCut', u'ROE', u'ROEW', u'ROECut', u'ROECutW', u'TAssets', u'TEquity', u'NAsset', u'NAssetPS']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','itemCD': 'str','item': 'str','chng': 'str','chngCause': 'str','oprFinExp': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def FdmtEFJLGet(secID = "", ticker = "", tickerB = "", beginDate = "", endDate = "", publishDateBegin = "", publishDateEnd = "", field = "", pandas = "1"):
    """
    获取上市公司披露的公告中的预期下一报告期收入、净利润、基本每股收益及其幅度变化等数据。（若上市公司同时发行债券等其他证券，也可通过其他证券代码查询预告数据）
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'000002.XSHE',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'000002',可以是列表,secID、ticker至少选择一个
    :param tickerB: 交易代码,如'900940',可以是列表,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMMDD”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMMDD”,可空
    :param publishDateBegin: 证券交易所披露的信息发布日期,起始时间,如‘20130812’,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期,结束时间,默认为当前日期,如‘20140812’,可空
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
    requestString.append('/api/listedCorp/getFdmtEFJL.csv?ispandas=1&') 
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
    requestString.append("&tickerB=")
    if hasattr(tickerB,'__iter__') and not isinstance(tickerB, str):
        if len(tickerB) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerB
            requestString.append(None)
        else:
            requestString.append(','.join(tickerB))
    else:
        requestString.append(tickerB)
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
    if not isinstance(publishDateBegin, str) and not isinstance(publishDateBegin, unicode):
        publishDateBegin = str(publishDateBegin)

    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    if not isinstance(publishDateEnd, str) and not isinstance(publishDateEnd, unicode):
        publishDateEnd = str(publishDateEnd)

    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'FdmtEFJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'FdmtEFJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'tickerA', u'AsecShortName', u'tickerB', u'BsecShortName', u'infoSource', u'publishDate', u'endDate', u'chng', u'chngCause', u'serial_Num', u'forecastType', u'summary', u'incomeLast', u'NIncomeChgrLL', u'NIncomeChgrUPL', u'EPSLast', u'expEPSLL', u'expEPSUPL', u'forecastCont', u'forecastReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','tickerA': 'str','AsecShortName': 'str','tickerB': 'str','BsecShortName': 'str','infoSource': 'str','chng': 'str','chngCause': 'str','forecastType': 'str','summary': 'str','forecastCont': 'str','forecastReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquJLGet(secID = "", ticker = "", exchangeCD = "", listSectorCD = "", equTypeCD = "", listStatus = "", field = "", pandas = "1"):
    """
    获取股票的基本信息，包含股票交易代码及其简称、股票类型、上市状态、上市板块、上市日期等；上市状态为最新数据，不显示历史变动信息。
    
    :param secID: 一只或多只证券代码，用,分隔，格式是“数字.交易所代码”，如000001.XSHE。如果为空，则为全部证券。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param ticker: 一只或多只股票代码，用,分隔，如000001,000002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param exchangeCD: 交易市场。例如，XSHG-上海证券交易所；XSHE-深圳证券交易所。对应DataAPI.SysCodeGet.codeTypeID=10002。,可以是列表,secID、ticker、exchangeCD至少选择一个
    :param listSectorCD: 上市板块。例如，1-主板；2-创业板；3-中小板。对应DataAPI.SysCodeGet.codeTypeID=10016。,可以是列表,可空
    :param equTypeCD: 股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
    :param listStatus: 股票上市状态，可选择上市，暂停，退市等。,可以是列表,可空
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
    requestString.append('/api/listedCorp/getEquJL.csv?ispandas=1&') 
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
    requestString.append("&listStatus=")
    if hasattr(listStatus,'__iter__') and not isinstance(listStatus, str):
        if len(listStatus) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = listStatus
            requestString.append(None)
        else:
            requestString.append(','.join(listStatus))
    else:
        requestString.append(listStatus)
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
            api_base.handle_error(csvString, 'EquJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'equTypeCD', u'listSectorCD', u'cnSpell', u'secShortName', u'secFullName', u'partyID', u'listDate', u'listStatus', u'delistDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchangeCD': 'str','equTypeCD': 'str','listSectorCD': 'str','cnSpell': 'str','secShortName': 'str','secFullName': 'str','listStatus': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquIPOpJLGet(partyID = "", aticker = "", bticker = "", equTypeCD = "", field = "", pandas = "1"):
    """
    获取股票首次公开发行上市招股意向书的基本信息，包含是否通过核准。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquIPOpJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquIPOpJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquIPOpJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'seq', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'csrcProcess', u'equTypeCD', u'intPublishDate', u'proPublishDate', u'parValueRmb', u'onlineIssueBeginDate', u'onlineIssueEndDate', u'offlineIssueBeginDate', u'offlineIssueEndDate', u'applyCodeOnlineIssue', u'applyAbbrOnlineIssue', u'issueModeCD', u'underwritingModeCD', u'sponsor']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','csrcProcess': 'str','equTypeCD': 'str','applyCodeOnlineIssue': 'str','applyAbbrOnlineIssue': 'str','issueModeCD': 'str','underwritingModeCD': 'str','sponsor': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquIPOrJLGet(partyID = "", aticker = "", bticker = "", equTypeCD = "", listDateStart = "", listDateEnd = "", field = "", pandas = "1"):
    """
    获取股票首次公开发行上市结果的基本信息。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
    :param equTypeCD: 发行股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
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
    requestString.append('/api/listedCorp/getEquIPOrJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
    try:
        listDateStart = listDateStart.strftime('%Y%m%d')
    except:
        listDateStart = listDateStart.replace('-', '')
    requestString.append("&listDateStart=%s"%(listDateStart))
    try:
        listDateEnd = listDateEnd.strftime('%Y%m%d')
    except:
        listDateEnd = listDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquIPOrJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquIPOrJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'pSeq', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'equTypeCD', u'listPublishDate', u'listDate', u'issuePrice', u'currencyCD', u'issueShares', u'newIssueShares', u'transShares', u'sharesBfIssue', u'sharesAfIssue', u'issueRaiseCap', u'newIssueRaiseCap', u'oldShareRaiseCap', u'issueCost', u'issueCostPerShare', u'onlineIssueLottoRatio', u'issueSharesOnline', u'issueSharesOffline', u'issuePED']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','equTypeCD': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquIPOmJLGet(secID = "", ticker = "", exchangeCD = "", listDateStart = "", listDateEnd = "", field = "", pandas = "1"):
    """
    获取股票首次公开发行上市首日的市场表现，包括开盘价、最高价、最低价、收盘价、涨幅以及成交量等数据。
    
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
    requestString.append('/api/listedCorp/getEquIPOmJL.csv?ispandas=1&') 
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
    try:
        listDateStart = listDateStart.strftime('%Y%m%d')
    except:
        listDateStart = listDateStart.replace('-', '')
    requestString.append("&listDateStart=%s"%(listDateStart))
    try:
        listDateEnd = listDateEnd.strftime('%Y%m%d')
    except:
        listDateEnd = listDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquIPOmJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquIPOmJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchangeCD', u'secShortName', u'listDate', u'firstDayVol', u'firstDayOpenPrice', u'firstDayHighestPrice', u'firstDayLowestPrice', u'firstDayClosePrice', u'firstDayTurnoverVol', u'firstDayTurnoverVal', u'firstDayChgPct']
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

def EquDivJLGet(partyID = "", Aticker = "", Bticker = "", endDate = "", endDateStart = "", endDateEnd = "", eventProcessCD = "", equTypeCD = "", field = "", pandas = "1"):
    """
    获取股票历次分红(派现、送股、转增股)的基本信息，包含历次分红预案的内容、实施进展情况、历史宣告分红次数以及股改分红情况。
    
    :param partyID: 一个上市公司的公司代码。,partyID、Aticker、Bticker、endDate至少选择一个
    :param Aticker: 一只上市公司A股股票代码，如000001。,partyID、Aticker、Bticker、endDate至少选择一个
    :param Bticker: 一只上市公司B股股票代码，如200002。,partyID、Aticker、Bticker、endDate至少选择一个
    :param endDate: 可输入报告期截止日，输入格式“YYYYMMDD”，如“20141231”，获取当期报告期所有股票分红信息,partyID、Aticker、Bticker、endDate至少选择一个
    :param endDateStart: 开始日期，输入格式YYYYMMDD，输入开始日期和截止日期，可获取期间报告期分红信息,可空
    :param endDateEnd: 截止日期，输入格式YYYYMMDD，输入开始日期和截止日期，可获取期间报告期分红信息,可空
    :param eventProcessCD: 事件进程。例如，1-董事预案；3-股东大会否决。对应DataAPI.SysCodeGet.codeTypeID=20001。,可以是列表,可空
    :param equTypeCD: 分红股票类别。例如，0201010201-A股；0201010202-B股。对应DataAPI.SysCodeGet.codeTypeID=20010。,可以是列表,可空
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
    requestString.append('/api/listedCorp/getEquDivJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(Aticker, str) and not isinstance(Aticker, unicode):
        Aticker = str(Aticker)

    requestString.append("&Aticker=%s"%(Aticker))
    if not isinstance(Bticker, str) and not isinstance(Bticker, unicode):
        Bticker = str(Bticker)

    requestString.append("&Bticker=%s"%(Bticker))
    try:
        endDate = endDate.strftime('%Y%m%d')
    except:
        endDate = endDate.replace('-', '')
    requestString.append("&endDate=%s"%(endDate))
    try:
        endDateStart = endDateStart.strftime('%Y%m%d')
    except:
        endDateStart = endDateStart.replace('-', '')
    requestString.append("&endDateStart=%s"%(endDateStart))
    try:
        endDateEnd = endDateEnd.strftime('%Y%m%d')
    except:
        endDateEnd = endDateEnd.replace('-', '')
    requestString.append("&endDateEnd=%s"%(endDateEnd))
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
            api_base.handle_error(csvString, 'EquDivJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquDivJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'partyID', u'Aticker', u'Bticker', u'AsecShortName', u'BsecShortName', u'endDate', u'equTypeCD', u'boPublishDate', u'imPublishDate', u'eventProcessCD', u'perCashDiv', u'perCashDivAfTax', u'currencyCD', u'perShareDivRatio', u'perShareTransRatio', u'divObject', u'recordDate', u'exDivDate', u'bLastTradeDate', u'payCashDate', u'bonusShareListDate', u'transShareListDate', u'baseShares']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','Aticker': 'str','Bticker': 'str','AsecShortName': 'str','BsecShortName': 'str','equTypeCD': 'str','eventProcessCD': 'str','currencyCD': 'str','divObject': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquAllotdJLGet(partyID = "", aticker = "", bticker = "", field = "", pandas = "1"):
    """
    获取公司历次配股获批后，公布的配股说明说的基本信息。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquAllotdJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquAllotdJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquAllotdJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'seq', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'proPublishDate', u'baseDate', u'baseShares', u'allotCodeA', u'allotCodeB', u'allotAbbrA', u'allotAbbrB', u'payBeginDate', u'payEndDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','allotCodeA': 'str','allotCodeB': 'str','allotAbbrA': 'str','allotAbbrB': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquAllotrJLGet(partyID = "", aticker = "", bticker = "", field = "", pandas = "1"):
    """
    获取公司历次配股实施的结果，包括配股价、配股比例、配股数量等信息。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquAllotrJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquAllotrJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquAllotrJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'dSeq', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'allotmentPrice', u'allotFrPrice', u'currencyCD', u'allotmentRatio', u'recordDateA', u'recordDateB', u'exRightsDateA', u'exRightsDateB', u'listDate', u'baseShares', u'allotShares', u'sharesBfAllot', u'sharesAfAllot', u'raiseCap', u'allotCost']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquSPOpJLGet(partyID = "", aticker = "", bticker = "", eventProcessCD = "", spoTypeCD = "", equTypeCD = "", changeTypeCD = "", field = "", pandas = "1"):
    """
    获取公司历次增发新股方案以及批准情况。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
    :param eventProcessCD: 事件进程。例如，1-董事会预案；2-股东大会通过；3-股东大会否决。对应DataAPI.SysCodeGet.codeTypeID=20001。,可以是列表,可空
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
    requestString.append('/api/listedCorp/getEquSPOpJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquSPOpJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquSPOpJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'iniPublishDate', u'issueSharesPul', u'issueSharesPll', u'shcVldEndDate', u'csrcVldEndDate', u'changeTypeCD', u'eventProcessCD', u'spoTypeCD', u'equTypeCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','changeTypeCD': 'str','eventProcessCD': 'str','spoTypeCD': 'str','equTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquSPOdJLGet(partyID = "", aticker = "", bticker = "", field = "", pandas = "1"):
    """
    获取公司公布的增发说明书或者增发实施情况的信息。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquSPOdJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquSPOdJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquSPOdJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'pSeq', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'proPublishDate', u'pIssuePrice', u'currency', u'pIssueShares', u'recordDate', u'exRightsDate', u'allotmentRatio']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','currency': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquSPOrJLGet(partyID = "", aticker = "", bticker = "", equTypeCD = "", field = "", pandas = "1"):
    """
    获取历次增发实施结果信息，包括发行价、发行量、发行费用等数据。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquSPOrJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquSPOrJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquSPOrJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'pSeq', u'dSeq', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'equTypeCD', u'listPublishDate', u'issuePrice', u'currency', u'issueShares', u'issueRaiseCap', u'sharesBfSpo', u'sharesAfSpo', u'tradeShares', u'listDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','equTypeCD': 'str','currency': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquRefJLGet(partyID = "", aticker = "", bticker = "", field = "", pandas = "1"):
    """
    获取公司股权分置改革的相关信息，包括实施进程及具体的支付对价情况。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquRefJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquRefJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquRefJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'iniPublishDate', u'eventProcess', u'imPublishDate', u'recordDate', u'changeDate', u'af1stTradeDate', u'cshareListDate', u'bfShares', u'bfTradeShares', u'afShares', u'actPerShare', u'actPerCash', u'callWarrantRatio', u'putWarrantRatio', u'comTransShare', u'comCash', u'shareConsiRatio', u'shareConsi']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','eventProcess': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquShareJLGet(partyID = "", aticker = "", bticker = "", changeTypeCD = "", changeDateStart = "", changeDateEnd = "", field = "", pandas = "1"):
    """
    获取上市公司最新股本结构及历次股本各部分变动情况数据。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquShareJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
    try:
        changeDateStart = changeDateStart.strftime('%Y%m%d')
    except:
        changeDateStart = changeDateStart.replace('-', '')
    requestString.append("&changeDateStart=%s"%(changeDateStart))
    try:
        changeDateEnd = changeDateEnd.strftime('%Y%m%d')
    except:
        changeDateEnd = changeDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquShareJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquShareJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'publishDate', u'changeDate', u'totalShares', u'bShares', u'hShares', u'sShares', u'nShares', u'floatShares', u'floatA', u'nonrestFloatA', u'restShares', u'nonListedShares', u'changeTypeCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','changeTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMsChanJLGet(partyID = "", aticker = "", bticker = "", relationship = "", changePctLl = "", changePctUl = "", field = "", pandas = "1"):
    """
    获取上市公司高管及其亲属买卖所在公司股份的情况，包括持股人与上市公司高管关系以及变动比例。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquMsChanJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquMsChanJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMsChanJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'reportDate', u'changeDate', u'managerName', u'position', u'shName', u'relationship', u'shareChar', u'sharesChange', u'avgPrice', u'changePct', u'sharesBfChange', u'sharesAfChange', u'changeReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','managerName': 'str','position': 'str','shName': 'str','relationship': 'str','shareChar': 'str','changeReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMainshFJLGet(partyID = "", aticker = "", bticker = "", shRank = "", shareCharType = "", endDateStart = "", endDateEnd = "", field = "", pandas = "1"):
    """
    获取公司十大流通股东历次变动记录，包括主要股东构成及持股数量比例等明细信息。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
    :param shRank: 十大流通股东持股排名，如输入1，得到第一大流通股东信息。,可以是列表,可空
    :param shareCharType: 股份性质类别。例如, 0101-流通A股；0102-流通B股。对应DataAPI.SysCodeGet.codeTypeID=20015。,可以是列表,可空
    :param endDateStart: 本次披露的十大流通股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
    :param endDateEnd: 本次披露的十大流通股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
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
    requestString.append('/api/listedCorp/getEquMainshFJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
    try:
        endDateStart = endDateStart.strftime('%Y%m%d')
    except:
        endDateStart = endDateStart.replace('-', '')
    requestString.append("&endDateStart=%s"%(endDateStart))
    try:
        endDateEnd = endDateEnd.strftime('%Y%m%d')
    except:
        endDateEnd = endDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquMainshFJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMainshFJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'endDate', u'shRank', u'shName', u'holdVol', u'holdPct', u'shareCharType', u'shareCharVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','shName': 'str','shareCharType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMainshJLGet(partyID = "", aticker = "", bticker = "", shareCharType = "", shRank = "", endDateStart = "", endDateEnd = "", field = "", pandas = "1"):
    """
    获取公司十大股东历次变动记录，包括主要股东构成及持股数量比例等明细信息。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquMainshJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
    try:
        endDateStart = endDateStart.strftime('%Y%m%d')
    except:
        endDateStart = endDateStart.replace('-', '')
    requestString.append("&endDateStart=%s"%(endDateStart))
    try:
        endDateEnd = endDateEnd.strftime('%Y%m%d')
    except:
        endDateEnd = endDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquMainshJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMainshJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'endDate', u'shRank', u'shName', u'holdVol', u'holdPct', u'shareCharType', u'shareCharVol']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','shName': 'str','shareCharType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMainshFdJLGet(partyID = "", aticker = "", bticker = "", shRank = "", changeTypeCD = "", endDateStart = "", endDateEnd = "", field = "", pandas = "1"):
    """
    获取公司十大流通股东历次变动记录，以包括持股类别明细情况。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
    :param shRank: 十大流通股东持股排名，如输入1，得到第一大流通股东信息。,可以是列表,可空
    :param changeTypeCD: 变动原因类别。例如，01-未变动；02-变动。对应DataAPI.SysCodeGet.codeTypeID=20005。,可以是列表,可空
    :param endDateStart: 本次披露的十大流通股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
    :param endDateEnd: 本次披露的十大流通股东统计截止日，输入格式“YYYYMMDD”，如输入起始日期"20130101"，截止日期"20131231"，可以查询到2013年期间披露的所有十大股东名单,可空
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
    requestString.append('/api/listedCorp/getEquMainshFdJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
    try:
        endDateStart = endDateStart.strftime('%Y%m%d')
    except:
        endDateStart = endDateStart.replace('-', '')
    requestString.append("&endDateStart=%s"%(endDateStart))
    try:
        endDateEnd = endDateEnd.strftime('%Y%m%d')
    except:
        endDateEnd = endDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquMainshFdJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMainshFdJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'publishDate', u'endDate', u'shRank', u'shName', u'holdVol', u'holdPct', u'shTypeDesc', u'holdRestVol', u'pleFrzCus', u'totShPct', u'changeTypeCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','shName': 'str','shTypeDesc': 'str','changeTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquMainshdJLGet(partyID = "", aticker = "", bticker = "", shRank = "", changeTypeCD = "", endDateStart = "", endDateEnd = "", field = "", pandas = "1"):
    """
    获取公司十大股东历次变动记录，以包括持股类别明细情况。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
    :param shRank: 十大股东持股排名，如输入1，得到第一大股东信息。,可以是列表,可空
    :param changeTypeCD: 变动原因类别。例如，01-未变动；02-变动。对应DataAPI.SysCodeGet.codeTypeID=20005。,可以是列表,可空
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
    requestString.append('/api/listedCorp/getEquMainshdJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
    try:
        endDateStart = endDateStart.strftime('%Y%m%d')
    except:
        endDateStart = endDateStart.replace('-', '')
    requestString.append("&endDateStart=%s"%(endDateStart))
    try:
        endDateEnd = endDateEnd.strftime('%Y%m%d')
    except:
        endDateEnd = endDateEnd.replace('-', '')
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
            api_base.handle_error(csvString, 'EquMainshdJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquMainshdJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'publishDate', u'endDate', u'shRank', u'shName', u'holdVol', u'holdPct', u'shTypeDesc', u'holdRestVol', u'pleFrzCus', u'totShPct', u'changeTypeCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','shName': 'str','shTypeDesc': 'str','changeTypeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquSuitJLGet(partyID, eventTypeCD = "", field = "", pandas = "1"):
    """
    获取公司诉讼仲裁等重大事项描述说明。
    
    :param partyID: 一个上市公司的公司代码。
    :param eventTypeCD: 行为事件类别。例如，0901-诉讼；0902-仲裁。对应DataAPI.SysCodeGet.codeTypeID=20002。,可以是列表,可空
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
    requestString.append('/api/listedCorp/getEquSuitJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    requestString.append("&eventTypeCD=")
    if hasattr(eventTypeCD,'__iter__') and not isinstance(eventTypeCD, str):
        if len(eventTypeCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = eventTypeCD
            requestString.append(None)
        else:
            requestString.append(','.join(eventTypeCD))
    else:
        requestString.append(eventTypeCD)
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
            api_base.handle_error(csvString, 'EquSuitJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquSuitJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'eventNum', u'partyID', u'publishDate', u'eventProcess', u'suitDate', u'cause', u'eventTypeCD', u'object', u'suitSum', u'currencyCD', u'judgOrg1st', u'judgOrgID1st', u'judgDate1st', u'judgResult1st', u'appelCD', u'appelName', u'appelID', u'appeals', u'judgOrg2nd', u'judgOrgID2nd', u'judgDate2nd', u'judgResult2nd', u'judgExe', u'complName', u'complID', u'complCD', u'complRelationshipCD', u'repaySum', u'suitCost']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'eventNum': 'str','eventProcess': 'str','suitDate': 'str','cause': 'str','eventTypeCD': 'str','object': 'str','currencyCD': 'str','judgOrg1st': 'str','judgResult1st': 'str','appelCD': 'str','appelName': 'str','appeals': 'str','judgOrg2nd': 'str','judgResult2nd': 'str','judgExe': 'str','complName': 'str','complCD': 'str','complRelationshipCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def EquPunishJLGet(partyID = "", aticker = "", bticker = "", punishReason = "", punishType = "", field = "", pandas = "1"):
    """
    获取公司违规受处罚事项描述说明。
    
    :param partyID: 一个上市公司的公司代码。,partyID、aticker、bticker至少选择一个
    :param aticker: 一只上市公司A股股票代码，如000001。,partyID、aticker、bticker至少选择一个
    :param bticker: 一只上市公司B股股票代码，如200002。,partyID、aticker、bticker至少选择一个
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
    requestString.append('/api/listedCorp/getEquPunishJL.csv?ispandas=1&') 
    if not isinstance(partyID, str) and not isinstance(partyID, unicode):
        partyID = str(partyID)

    requestString.append("partyID=%s"%(partyID))
    if not isinstance(aticker, str) and not isinstance(aticker, unicode):
        aticker = str(aticker)

    requestString.append("&aticker=%s"%(aticker))
    if not isinstance(bticker, str) and not isinstance(bticker, unicode):
        bticker = str(bticker)

    requestString.append("&bticker=%s"%(bticker))
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
            api_base.handle_error(csvString, 'EquPunishJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'EquPunishJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyID', u'aticker', u'bticker', u'aSecShortName', u'bSecShortName', u'publishDate', u'punishReason', u'punishType', u'punishOrg', u'irregBeginYear', u'irregEndYear']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'aticker': 'str','bticker': 'str','aSecShortName': 'str','bSecShortName': 'str','punishReason': 'str','punishType': 'str','punishOrg': 'str','irregBeginYear': 'str','irregEndYear': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktHKEqudJLGet(secID = "", ticker = "", startDate = "", endDate = "", field = "", pandas = "1"):
    """
    本表记录港股日行情信息。包括交易日期、开盘价、收盘价、成交量、委托买卖情况、涨跌等重要行情指标，历史追溯至1986年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'00001'，可获取到secID'00001.XHKG'后，在此输入'00001.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 证券代码,可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/HKsec/getMktHKEqudJL.csv?ispandas=1&') 
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
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
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
            api_base.handle_error(csvString, 'MktHKEqudJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktHKEqudJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverValue', u'turnoverVol', u'chg', u'chgPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktHKEqumJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表为港股的股票月行情表;暂时不考虑复权的情况。目前只计算1990年以后的数据。包含月首个交易日,月最后交易日,月交易天数,月前收盘价,月开盘价,月最高价,月最高价日,月最低价,月最低价日,月收盘价,月最高收盘价,月最高收盘价日,月最低收盘价,月最低收盘价日,月成交均价等，历史追溯至1990年，每月更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'00001'，可获取到secID'00001.XHKG'后，在此输入'00001.XHKG'
    :param startDate: 起始日期，输入格式为yyyymmdd(每月月末）,可空
    :param finishDate: 结束日期，输入格式为yyyymmdd(每月月末）,可空
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
    requestString.append('/api/HKsec/getMktHKEqumJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, 'MktHKEqumJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktHKEqumJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'firstDateM', u'lastDateM', u'numDaysM', u'preClosePriceM', u'openPriceM', u'highestPriceM', u'dayHigh', u'lowestPriceM', u'dayLow', u'closePriceM', u'highClosePriceM', u'dayHcpM', u'lowClosePriceM', u'dayLcpM', u'avgPriceM', u'rangePctM', u'chgM', u'chgPctM', u'logReturnM', u'turnoverVolM', u'turnoverValueM', u'turnoverRateM']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktHKEquwJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表为港股的股票周行情表，暂时不考虑复权的情况。目前只计算1990年以后的数据。包含周前收盘价,周开盘价,周最高价,周最高价日,周最低价,周最低价日,周收盘价,周最高收盘价,周最高收盘价日,周最低收盘价,周最低收盘价日,周成交均价,周振幅,周涨跌,周涨跌幅等数据。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'00001'，可获取到secID'00001.XHKG'后，在此输入'00001.XHKG'
    :param startDate: 起始日期，输入格式为yyyymmdd（每周最后一个交易日）,可空
    :param finishDate: 结束日期，输入格式为yyyymmdd（每周最后一个交易日）,可空
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
    requestString.append('/api/HKsec/getMktHKEquwJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, 'MktHKEquwJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktHKEquwJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'firstDateW', u'lastDateW', u'numDaysW', u'preClosePriceW', u'openPriceW', u'highestPriceW', u'dayHigh', u'lowestPriceW', u'dayLow', u'closePriceW', u'highClosePriceW', u'dayHcpW', u'lowClosePriceW', u'dayLcpW', u'avgPriceW', u'rangePctW', u'chgW', u'chgPctW', u'logReturnW', u'turnoverVolW', u'turnoverValueW', u'turnoverRateW']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktHKEquyJLGet(secID, beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    本表为港股的股票年行情表;暂时不考虑复权的情况。目前只计算1990年以后的数据。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'00001'，可获取到secID'00001.XHKG'后，在此输入'00001.XHKG'
    :param beginDate: 起始日期，输入格式为yyyymmdd(每年年末）,可空
    :param endDate: 结束日期，输入格式为yyyymmdd(每年年末）,可空
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
    requestString.append('/api/HKsec/getMktHKEquyJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
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
            api_base.handle_error(csvString, 'MktHKEquyJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktHKEquyJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'firstDateY', u'lastDateY', u'numDaysY', u'preClosePriceY', u'openPriceY', u'highestPriceY', u'dayHigh', u'lowestPriceY', u'dayLow', u'closePriceY', u'highClosePriceY', u'dayHcpY', u'lowClosePriceY', u'dayLcpY', u'avgPriceY', u'rangePctY', u'chgY', u'chgPctY', u'logReturnY', u'turnoverVolY', u'turnoverValueY', u'turnoverRateY']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtFBSJLGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股金融类上市公司资产负债表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00011.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00011',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMM”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMM”,可空
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
    requestString.append('/api/HKsec/getHKFdmtFBSJL.csv?ispandas=1&') 
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
        publishDateBegin = publishDateBegin.strftime('%Y%m%d')
    except:
        publishDateBegin = publishDateBegin.replace('-', '')
    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    try:
        publishDateEnd = publishDateEnd.strftime('%Y%m%d')
    except:
        publishDateEnd = publishDateEnd.replace('-', '')
    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'HKFdmtFBSJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtFBSJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'repTypeCD', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'exRate', u'perFStatBLCD', u'cashSTFun', u'colOthBK', u'TDeposFrOthBFIA', u'tradeBill', u'depositCard', u'owesCertiGovHK', u'tradingFA', u'NtradingFA', u'fairValueFA', u'derivAssets', u'BKLoanOthAcc', u'clientLoanOthAcc', u'availForSaleFA', u'FAInvest', u'HIMInvest', u'assocComEquity', u'intanAssets', u'fixedAssets', u'othAssets', u'TAssets', u'HKDollar', u'deposFrOthBFI', u'depos', u'transOthBK', u'TDeposFrOthBFIL', u'issuedDebts', u'tradingFL', u'fairValueFL', u'derivLiab', u'SBNLiab', u'othLiab', u'TLiab', u'paidInCapital', u'stockPremium', u'capitalReser', u'othReser', u'holdProfit', u'TReser', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TShEquityAndTLiab', u'CTLAndPMI']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','repTypeCD': 'str','fiscalPeriod': 'str','currencyCD': 'str','perFStatBLCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtNFBSJLGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股非金融类上市公司资产负债表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00001.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMM”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMM”,可空
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
    requestString.append('/api/HKsec/getHKFdmtNFBSJL.csv?ispandas=1&') 
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
        publishDateBegin = publishDateBegin.strftime('%Y%m%d')
    except:
        publishDateBegin = publishDateBegin.replace('-', '')
    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    try:
        publishDateEnd = publishDateEnd.strftime('%Y%m%d')
    except:
        publishDateEnd = publishDateEnd.replace('-', '')
    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'HKFdmtNFBSJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtNFBSJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'repTypeCD', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'exRate', u'perFStatBLCD', u'fixedAssets', u'invest', u'intanAssets', u'othNCA', u'TNCA', u'cash', u'FA', u'tradeAR', u'inventories', u'othCA', u'TCA', u'TAssets', u'tradeAP', u'STBorr', u'othCL', u'TCL', u'NCA', u'TAssetsLessTCL', u'LTBorr', u'othNCL', u'TNCL', u'NAL', u'paidInCapital', u'stockPremium', u'capitalReser', u'othReser', u'holdProfit', u'TReser', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TLiabMinorityInt', u'TLiab', u'CTLAndPMI']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','repTypeCD': 'str','fiscalPeriod': 'str','currencyCD': 'str','perFStatBLCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtFISJLGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股金融类上市公司利润表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00011.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00011',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMM”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMM”,可空
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
    requestString.append('/api/HKsec/getHKFdmtFISJL.csv?ispandas=1&') 
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
        publishDateBegin = publishDateBegin.strftime('%Y%m%d')
    except:
        publishDateBegin = publishDateBegin.replace('-', '')
    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    try:
        publishDateEnd = publishDateEnd.strftime('%Y%m%d')
    except:
        publishDateEnd = publishDateEnd.replace('-', '')
    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'HKFdmtFISJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtFISJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'repTypeCD', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'exRate', u'perFStatBLCD', u'intIncome', u'intExp', u'NIntIncome', u'commisIncome', u'commisExp', u'NCommisIncome', u'NTradingIncome', u'NPremEarned', u'othOperRev', u'TOthOperRev', u'premEarned', u'NInsurCL', u'befPrepOperProfit', u'devalAccBadDebt', u'TExp', u'operateProfit', u'NOperateSpec', u'comAssocProfit', u'TProfit', u'tax', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'PShDiv', u'CPPYCapSec', u'shareHoldProfit', u'othComprIncome', u'TComprIncome', u'CShDiv', u'basicEPSAnRpt', u'basicEPSA', u'dilutedEPSA', u'DPSAP', u'conBusBefTaxProf', u'conBusTax', u'conBusNIncome', u'NConBusNIncome', u'basicEPSConBusA', u'basicEPSNConBusA', u'dilutedEPSConBusA', u'dilutedEPSNConBusA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','repTypeCD': 'str','fiscalPeriod': 'str','currencyCD': 'str','perFStatBLCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtNFISJLGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股非金融类上市公司利润表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00001.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMM”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMM”,可空
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
    requestString.append('/api/HKsec/getHKFdmtNFISJL.csv?ispandas=1&') 
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
        publishDateBegin = publishDateBegin.strftime('%Y%m%d')
    except:
        publishDateBegin = publishDateBegin.replace('-', '')
    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    try:
        publishDateEnd = publishDateEnd.strftime('%Y%m%d')
    except:
        publishDateEnd = publishDateEnd.replace('-', '')
    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'HKFdmtNFISJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtNFISJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'repTypeCD', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'exRate', u'perFStatBLCD', u'revenue', u'salesCosts', u'grossIncome', u'othOperRev', u'TRevenue', u'sellExp', u'adminExp', u'othOperExp', u'TExp', u'RD', u'devalAccBadDebt', u'operProfBefFC', u'finanCost', u'operateProfit', u'NOperateSpec', u'comAssocProfit', u'TProfit', u'tax', u'taxExcIncomeTax', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'shareHoldProfit', u'othComprIncome', u'TComprIncome', u'CShDiv', u'basicEPSAnRpt', u'basicEPSA', u'dilutedEPSA', u'DPSAP', u'depAmor', u'conBusBefTaxProf', u'conBusTax', u'conBusNIncome', u'NConBusNIncome', u'basicEPSConBusA', u'basicEPSNConBusA', u'dilutedEPSConBusA', u'dilutedEPSNConBusA']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','repTypeCD': 'str','fiscalPeriod': 'str','currencyCD': 'str','perFStatBLCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtCFJLGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股上市公司现金流量表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00001.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param publishDateBegin: 证券交易所披露的信息发布日期，起始时间，输入格式“YYYYMMDD”,可空
    :param publishDateEnd: 证券交易所披露的信息发布日期，结束时间，默认为当前日期，输入格式“YYYYMMDD”,可空
    :param beginDate: 会计期间截止日期，起始时间，输入格式“YYYYMM”,可空
    :param endDate: 会计期间截止日期，结束时间，输入格式“YYYYMM”,可空
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
    requestString.append('/api/HKsec/getHKFdmtCFJL.csv?ispandas=1&') 
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
        publishDateBegin = publishDateBegin.strftime('%Y%m%d')
    except:
        publishDateBegin = publishDateBegin.replace('-', '')
    requestString.append("&publishDateBegin=%s"%(publishDateBegin))
    try:
        publishDateEnd = publishDateEnd.strftime('%Y%m%d')
    except:
        publishDateEnd = publishDateEnd.replace('-', '')
    requestString.append("&publishDateEnd=%s"%(publishDateEnd))
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
            api_base.handle_error(csvString, 'HKFdmtCFJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtCFJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'exchangeCD', u'ticker', u'secShortName', u'repTypeCD', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'exRate', u'perFStatBLCD', u'NCFOperateA', u'CFOperateA', u'depAmor', u'receivInt', u'payableInt', u'receivDivFinan', u'disDivFinan', u'taxPRF', u'purFixAssets', u'investInc', u'dispFixAssets', u'investDec', u'specInvest', u'NCFFrInvestA', u'newLoan', u'refund', u'fixRateDebtFinan', u'EquityFinan', u'specFinan', u'RPFixRateDebt', u'NCFFrFinanA', u'NChangeInCash', u'NCEBegBal', u'forexEffects', u'NCEEndBal']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','exchangeCD': 'str','ticker': 'str','secShortName': 'str','repTypeCD': 'str','fiscalPeriod': 'str','currencyCD': 'str','perFStatBLCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKEquJLGet(secId = "", tickerSymbol = "", partyId = "", field = "", pandas = "1"):
    """
    香港股票代码基础信息，涵盖的信息包括股票代码，股票简称，股票拼音简称，公司统一代码，每手买卖单位，面值，面值货币统一编码，上市状态参数，上市板块，股票类型，SEDOL代码，ISIN代码，首发上市日，退市日期，交易货币代码
    
    :param secId: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表,secId、tickerSymbol、partyId至少选择一个
    :param tickerSymbol: 交易代码,如‘00001’,可以是列表,secId、tickerSymbol、partyId至少选择一个
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到,secId、tickerSymbol、partyId至少选择一个
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
    requestString.append('/api/HKsec/getHKEquJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    requestString.append("&tickerSymbol=")
    if hasattr(tickerSymbol,'__iter__') and not isinstance(tickerSymbol, str):
        if len(tickerSymbol) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = tickerSymbol
            requestString.append(None)
        else:
            requestString.append(','.join(tickerSymbol))
    else:
        requestString.append(tickerSymbol)
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("&partyId=%s"%(partyId))
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
            api_base.handle_error(csvString, 'HKEquJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKEquJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'partyId', u'tickerSymbol', u'secShortName', u'cnSpell', u'tradingUnit', u'parValue', u'parCurrencyCd', u'listStatusCd', u'listSectorCd', u'exchangeCd', u'equTypeCd', u'sedolCode', u'isinCode', u'IPODate', u'delistDate', u'transCurrCd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','tickerSymbol': 'str','secShortName': 'str','cnSpell': 'str','parCurrencyCd': 'str','listStatusCd': 'str','listSectorCd': 'str','exchangeCd': 'str','equTypeCd': 'str','sedolCode': 'str','isinCode': 'str','transCurrCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKAbbchJLGet(secId, changeDate = "", field = "", pandas = "1"):
    """
    香港股票简称变动表，涵盖股票代码，变动日期，公告日期，股票简称，股票拼音简称，股票英文简称，变动原因
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param changeDate: 变动日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKAbbchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        changeDate = changeDate.strftime('%Y%m%d')
    except:
        changeDate = changeDate.replace('-', '')
    requestString.append("&changeDate=%s"%(changeDate))
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
            api_base.handle_error(csvString, 'HKAbbchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKAbbchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'changeDate', u'annoDate', u'secShortName', u'cnSpell', u'secShortNameEn', u'changeReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','secShortName': 'str','cnSpell': 'str','secShortNameEn': 'str','changeReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKSsalJLGet(secId, tradeDate, field = "", pandas = "1"):
    """
    香港证券沽空资料，涵盖股票代码，交易日期，公告日期，沽空数量，沽空金额
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param tradeDate: 交易日，输入格式“YYYYMMDD”
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
    requestString.append('/api/HKsec/getHKSsalJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        tradeDate = tradeDate.strftime('%Y%m%d')
    except:
        tradeDate = tradeDate.replace('-', '')
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
            api_base.handle_error(csvString, 'HKSsalJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKSsalJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'tradeDate', u'annoDate', u'shortSalesSum', u'shortSalesAmt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKSbpJLGet(secId, initiInfoPubDate = "", field = "", pandas = "1"):
    """
    香港股票分拆、合并、并行资料，记录香港上市公司各股票的分拆、合并、重组及并行交易的相关信息，包括日期信息、方案信息、面值变动信息、交易单位变动信息、并行信息、零碎股信息的内容
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param initiInfoPubDate: 首次信息公布日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKSbpJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        initiInfoPubDate = initiInfoPubDate.strftime('%Y%m%d')
    except:
        initiInfoPubDate = initiInfoPubDate.replace('-', '')
    requestString.append("&initiInfoPubDate=%s"%(initiInfoPubDate))
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
            api_base.handle_error(csvString, 'HKSbpJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKSbpJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'id', u'initiInfoPubDate', u'dirAnnoDate', u'dirSignDate', u'mtDate', u'annoDate', u'chgEffectDate', u'expAnnoDate', u'processCd', u'remark', u'equTypeCd', u'stockIssueStateCd', u'assetRecombTypeCd', u'wrtOffBase', u'wrtOffParValue', u'parValueNew', u'parValueOld', u'parCurrencyCd', u'newTradingUnit', u'oldTradingUnit', u'combX', u'combY', u'splitX', u'splitY', u'issueShares', u'sharesAfEffect', u'tmpTicker', u'tmpStkName', u'tmpStkTradingUnit', u'tmpDateBegin', u'parDateBegin', u'parDateEnd', u'oddlotsTradeBegin', u'oddlotsTradeEnd', u'oddlotsAgent', u'oddlotsAgentAdd', u'oddlotsAgentTel']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','processCd': 'str','remark': 'str','equTypeCd': 'str','stockIssueStateCd': 'str','assetRecombTypeCd': 'str','parCurrencyCd': 'str','tmpTicker': 'str','tmpStkName': 'str','oddlotsAgent': 'str','oddlotsAgentAdd': 'str','oddlotsAgentTel': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKBbkJLGet(secId, buyBackDate = "", field = "", pandas = "1"):
    """
    香港公司回购资料，记录香港上市公司回购本公司发行的股票或权证的相关信息，如：回购日期、回购股数、回购最高价格\最低价格\平均价格、回购金额等
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param buyBackDate: 回购日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKBbkJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        buyBackDate = buyBackDate.strftime('%Y%m%d')
    except:
        buyBackDate = buyBackDate.replace('-', '')
    requestString.append("&buyBackDate=%s"%(buyBackDate))
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
            api_base.handle_error(csvString, 'HKBbkJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKBbkJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'buyBackDate', u'annoDate', u'buyBackSum', u'highestPrice', u'lowestPrice', u'avgPrice', u'buyBackMoney', u'cumulSum', u'cumulSumToTs', u'currencyCd', u'exchangeCd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','currencyCd': 'str','exchangeCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFinPJLGet(partyId, initiInfoPubDate = "", field = "", pandas = "1"):
    """
    香港公司融资计划资料，记录香港公司融资计划的相关信息，涵盖相关的重要日期，上市方式，发行类型发行对象，承销方式定量方式，公开发售，配售，供股的信息，发行价格，预计募集资金，预计发行费用，募资用途，未缴款供股权等信息
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param initiInfoPubDate: 首次信息发布日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKFinPJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        initiInfoPubDate = initiInfoPubDate.strftime('%Y%m%d')
    except:
        initiInfoPubDate = initiInfoPubDate.replace('-', '')
    requestString.append("&initiInfoPubDate=%s"%(initiInfoPubDate))
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
            api_base.handle_error(csvString, 'HKFinPJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFinPJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'fcgNum', u'initiInfoPubDate', u'prospPubDate', u'dirAnnoDate', u'dirSignDate', u'mtDate', u'ctctSignDate', u'expAnnoDate', u'tickerSymbol', u'listModeCd', u'listModeDesc', u'listSectorCd', u'issueTypeCd', u'issueObjectCd', u'underwModelCd', u'rationModelCd', u'issueVol', u'newShares', u'underwShareNum', u'pubOfferShareNum', u'pubOfferNshare', u'pubOfferStshare', u'staffPriAllot', u'allotShareNum', u'nshareAllot', u'stshareAllot', u'qedShPriAllot', u'rightIssueShares', u'bonusShares', u'oaoPlan', u'sharesBfList', u'offerRationX', u'offerRationY', u'highestIssuePrice', u'lowestIssuePrice', u'proceedsPlan', u'netPrpceedsPlan', u'issueCostPlan', u'oallotPrcdPlan', u'currencyCd', u'commisionRatio', u'parValue', u'parCurrencyCd', u'tradingUnit', u'capPur', u'nonpayRightCode', u'nonpayRightName', u'nonpayRitNameEn', u'npayRitTradeUnit', u'npayRitTradeBegin', u'npayRitTradeEnd', u'rightPayEnd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'tickerSymbol': 'str','listModeCd': 'str','listModeDesc': 'str','listSectorCd': 'str','issueTypeCd': 'str','issueObjectCd': 'str','underwModelCd': 'str','rationModelCd': 'str','currencyCd': 'str','parCurrencyCd': 'str','capPur': 'str','nonpayRightCode': 'str','nonpayRightName': 'str','nonpayRitNameEn': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFinRJLGet(partyId, issueResPubDate = "", field = "", pandas = "1"):
    """
    香港公司融资结果资料，记录香港公司融资结构的相关信息，涵盖发行总股数，新股数量，公开发售，配售，红股，超额配售，募资总额，募资净额，公开发售认购信息，职工优先配售信息，配售信息，合资格股东优先配售信息，有效认购数信息，不可撤回及认购不足数信息，包销未认购信息，承销商数目等
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param issueResPubDate: 上市结果公告日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKFinRJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        issueResPubDate = issueResPubDate.strftime('%Y%m%d')
    except:
        issueResPubDate = issueResPubDate.replace('-', '')
    requestString.append("&issueResPubDate=%s"%(issueResPubDate))
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
            api_base.handle_error(csvString, 'HKFinRJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFinRJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'fcgNum', u'relationFlag', u'issueResPubDate', u'issueVol', u'ratioInPrishas', u'ratioInPostshas', u'newShares', u'underwShareNum', u'pubOfferShareNum', u'pubOfferNshare', u'pubOfferStshare', u'staffPriAllot', u'allotShareNum', u'nshareAllot', u'stshareAllot', u'qedShPriAllot', u'rightIssueShares', u'bonusShares', u'overAllotOption', u'issueShareAfOao', u'issuePrice', u'issueCurrencyCd', u'totalProceeds', u'netPrpceeds', u'overAllotProceeds', u'overAllotNetPrcd', u'currencyCd', u'capPur', u'pubApplyUnit', u'pubApplyShares', u'pubApplyMutip', u'staffApplyUnit', u'staffApplyShares', u'staffApplyMutip', u'allotApplyUnit', u'allotApplyShares', u'allotApplyMutip', u'qedShApplyUnit', u'qedShApplyShares', u'qedShApplyMutip', u'validApplyShares', u'vapplyAllotShares', u'vapplyOallotShares', u'irbleUndertShares', u'subsRemainShares', u'underwBoughtVol', u'underwNum']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'issueCurrencyCd': 'str','currencyCd': 'str','capPur': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKOphyJLGet(partyId, initiInfoPubDate = "", field = "", pandas = "1"):
    """
    香港公司分红派息概括表，记录香港上市公司实物分红概貌信息，实物是指除现金、红股、认股权证外的分红标的，包含分红事件的总体日期进程、分红总数、分红描述说明等信息
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param initiInfoPubDate: 首次信息发布日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKOphyJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        initiInfoPubDate = initiInfoPubDate.strftime('%Y%m%d')
    except:
        initiInfoPubDate = initiInfoPubDate.replace('-', '')
    requestString.append("&initiInfoPubDate=%s"%(initiInfoPubDate))
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
            api_base.handle_error(csvString, 'HKOphyJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKOphyJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'divNum', u'endDate', u'processCd', u'fiscalReportDate', u'fiscalPeriodCd', u'initiInfoPubDate', u'dirAnnoDate', u'dirMeetingDate', u'dirConditionCd', u'divBase', u'divObjectCd', u'divRemark', u'smAnnoDate', u'smDate', u'effectAnnoDate', u'divChgCd', u'divChgAnnoDate', u'divChgRemark', u'lastTradeDay', u'exDate', u'reportDate', u'payDate', u'nonRegBegin', u'nonRegEnd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'processCd': 'str','fiscalPeriodCd': 'str','dirConditionCd': 'str','divObjectCd': 'str','divRemark': 'str','divChgCd': 'str','divChgRemark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKDTphyJLGet(partyId, field = "", pandas = "1"):
    """
    香港公司分红派息明细表，记录香港上市公司实物分红详细信息，如实物代码，实物名称，实物分配比例等信息
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
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
    requestString.append('/api/HKsec/getHKDTphyJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
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
            api_base.handle_error(csvString, 'HKDTphyJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKDTphyJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'divNum', u'relationFlag', u'phyNum', u'actualBonusShare', u'physicalDivRateX', u'physicalDivRateY', u'phySecId', u'phyTickerSymbol', u'phySecShortName', u'phyDesc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'phyTickerSymbol': 'str','phySecShortName': 'str','phyDesc': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKCdivJLGet(partyId, initiInfoPubDate = "", field = "", pandas = "1"):
    """
    香港公司现金分红表，记录香港上市公司现金分红的详细情况，包含现金分红的日期进程、分红总数、分红对象、分红描述说明、以股代息等信息
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param initiInfoPubDate: 首次信息发布日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKCdivJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        initiInfoPubDate = initiInfoPubDate.strftime('%Y%m%d')
    except:
        initiInfoPubDate = initiInfoPubDate.replace('-', '')
    requestString.append("&initiInfoPubDate=%s"%(initiInfoPubDate))
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
            api_base.handle_error(csvString, 'HKCdivJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKCdivJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'divNum', u'endDate', u'processCd', u'fiscalReportDate', u'fiscalPeriodCd', u'divTypeCd', u'initiInfoPubDate', u'dirAnnoDate', u'dirMeetingDate', u'divBase', u'divObjectCd', u'divRemark', u'smAnnoDate', u'smDate', u'effectAnnoDate', u'divChgCd', u'divChgAnnoDate', u'divChgRemark', u'lastTradeDay', u'exDate', u'reportDate', u'payDate', u'nonRegBegin', u'nonRegEnd', u'dps', u'specialDps', u'divCurrencyCd', u'cashDivTotal', u'cashDivCurrencyCd', u'convsBenchmarkCd', u'scripTypeCd', u'actualCashDiv', u'scripConvsPrice', u'convsCurrencyCd', u'scrrpShares', u'actualScripCash', u'actualScripShares', u'scripDivPoutDate', u'scripDivListDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'processCd': 'str','fiscalPeriodCd': 'str','divTypeCd': 'str','divObjectCd': 'str','divRemark': 'str','divChgCd': 'str','divChgRemark': 'str','divCurrencyCd': 'str','cashDivCurrencyCd': 'str','convsBenchmarkCd': 'str','scripTypeCd': 'str','convsCurrencyCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKBwdivJLGet(partyId, initiInfoPubDate = "", field = "", pandas = "1"):
    """
    香港公司送红股或红利认股证信息，记录香港上市公司送红股或红利认股证的分红概貌信息，包含分红事件的总体日期进程、送红股或红利认股证的配送比例、分红描述说明等信息
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param initiInfoPubDate: 首次信息发布日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKBwdivJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        initiInfoPubDate = initiInfoPubDate.strftime('%Y%m%d')
    except:
        initiInfoPubDate = initiInfoPubDate.replace('-', '')
    requestString.append("&initiInfoPubDate=%s"%(initiInfoPubDate))
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
            api_base.handle_error(csvString, 'HKBwdivJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKBwdivJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'divNum', u'endDate', u'processCd', u'fiscalReportDate', u'fiscalPeriodCd', u'initiInfoPubDate', u'dirAnnoDate', u'dirMeetingDate', u'divBase', u'divObjectCd', u'divRemark', u'smAnnoDate', u'smDate', u'effectAnnoDate', u'divChgCd', u'divChgAnnoDate', u'divChgRemark', u'lastTradeDay', u'exDate', u'reportDate', u'payDate', u'nonRegBegin', u'nonRegEnd', u'totalShareDiv', u'shareDivRateX', u'shareDivRateY', u'payOutDate', u'bonusShareLdate', u'totalWtsDiv', u'wtsDivRateX', u'wtsDivRateY', u'wtsDivPrice', u'priceCurrCd', u'wtsYear', u'wtsTradingUnit', u'subsSharesBegin', u'subsSharesEnd', u'wtsPayOutDate', u'wtsListDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'processCd': 'str','fiscalPeriodCd': 'str','divObjectCd': 'str','divRemark': 'str','divChgCd': 'str','divChgRemark': 'str','priceCurrCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKCodchJLGet(secId = "", ticker = "", partyId = "", changeDate = "", field = "", pandas = "1"):
    """
    香港公司股票代码变动信息，记录因公司首发上市、转板上市等原因引发股票代码变动的信息
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表,secId、ticker、partyId至少选择一个
    :param ticker: 交易代码，如‘00001’,可以是列表,secId、ticker、partyId至少选择一个
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到,secId、ticker、partyId至少选择一个
    :param changeDate: 变动日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKCodchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
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
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("&partyId=%s"%(partyId))
    try:
        changeDate = changeDate.strftime('%Y%m%d')
    except:
        changeDate = changeDate.replace('-', '')
    requestString.append("&changeDate=%s"%(changeDate))
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
            api_base.handle_error(csvString, 'HKCodchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKCodchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'partyId', u'changeDate', u'chgTypeCd', u'isinCode', u'sectorCd', u'sedolCode', u'equTypeCd', u'ticker', u'exchangeCd', u'currencyCd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','chgTypeCd': 'str','isinCode': 'str','sectorCd': 'str','sedolCode': 'str','equTypeCd': 'str','ticker': 'str','exchangeCd': 'str','currencyCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKTuchJLGet(secId, changeDate = "", field = "", pandas = "1"):
    """
    香港公司股票交易单位信息，记录股票每手买卖单位变动的信息
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param changeDate: 变动日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKTuchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        changeDate = changeDate.strftime('%Y%m%d')
    except:
        changeDate = changeDate.replace('-', '')
    requestString.append("&changeDate=%s"%(changeDate))
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
            api_base.handle_error(csvString, 'HKTuchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKTuchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'changeDate', u'tradingUnitNew', u'chgReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','chgReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKPvchJLGet(secId, changeDate = "", field = "", pandas = "1"):
    """
    香港公司每股面值变动信息，记录每股面值变动的信息
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param changeDate: 变动日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKPvchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        changeDate = changeDate.strftime('%Y%m%d')
    except:
        changeDate = changeDate.replace('-', '')
    requestString.append("&changeDate=%s"%(changeDate))
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
            api_base.handle_error(csvString, 'HKPvchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKPvchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'changeDate', u'parValueNew', u'parCurrencyCd', u'chgReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','parCurrencyCd': 'str','chgReason': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKStuchJLGet(secId, changeDate = "", field = "", pandas = "1"):
    """
    香港股票状态变动信息，记录所有香港交易所的上市公司发行股票的交易状态信息，包括是否正常上市，是否退市，日期和原因等内容
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param changeDate: 变动日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKStuchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        changeDate = changeDate.strftime('%Y%m%d')
    except:
        changeDate = changeDate.replace('-', '')
    requestString.append("&changeDate=%s"%(changeDate))
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
            api_base.handle_error(csvString, 'HKStuchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKStuchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'changeDate', u'chgReason']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKSasrchJLGet(suitSprLowLimit = "", suitSprCeiling = "", field = "", pandas = "1"):
    """
    香港股票挂单价位变动表，记录香港证券市场的各种证券品种在不同年代的挂单价位最小间隔范围变动情况
    
    :param suitSprLowLimit: 适用股价下限，有11种情况，‘0.01’‘0.25’‘0.50’‘10.00’‘20.00’‘100.00’‘200.00’‘500.00’‘1000.00’‘2000.00’‘5000.00’,suitSprLowLimit、suitSprCeiling至少选择一个
    :param suitSprCeiling: 适用股价上限，有11种情况，‘0.25’‘0.50’‘10.00’‘20.00’‘100.00’‘200.00’‘500.00’‘1000.00’‘2000.00’‘5000.00’‘9995.00’,suitSprLowLimit、suitSprCeiling至少选择一个
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
    requestString.append('/api/HKsec/getHKSasrchJL.csv?ispandas=1&') 
    if not isinstance(suitSprLowLimit, str) and not isinstance(suitSprLowLimit, unicode):
        suitSprLowLimit = str(suitSprLowLimit)

    requestString.append("suitSprLowLimit=%s"%(suitSprLowLimit))
    if not isinstance(suitSprCeiling, str) and not isinstance(suitSprCeiling, unicode):
        suitSprCeiling = str(suitSprCeiling)

    requestString.append("&suitSprCeiling=%s"%(suitSprCeiling))
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
            api_base.handle_error(csvString, 'HKSasrchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKSasrchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'equTypeCd', u'exchangeCd', u'changeDate', u'id', u'suitSprCeiling', u'suitSprLowLimit', u'prChgUnit']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'equTypeCd': 'str','exchangeCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKUwdtJLGet(partyId, field = "", pandas = "1"):
    """
    香港股票承配明细表，涵盖承配类型，承配配售股数，占本次配售股数比例，占发售后已发行股本比例
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
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
    requestString.append('/api/HKsec/getHKUwdtJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
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
            api_base.handle_error(csvString, 'HKUwdtJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKUwdtJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'fcgNum', u'retionFlag', u'underwTypeCd', u'underwNum', u'underwAllot', u'ratioInTotUnderw', u'ratioInPostshas']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'underwTypeCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKEqudtJLGet(partyId, changeDate = "", annoDate = "", field = "", pandas = "1"):
    """
    香港公司股本情况明细，记录香港上市公司的股本情况明细表,记录股本变动原因、股本变动后普通股、优先股、递延股的实收股本、法定股本、实收股数、法定股数等信息
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param changeDate: 变动日期，输入格式“YYYYMMDD”,可空
    :param annoDate: 公告日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKEqudtJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        changeDate = changeDate.strftime('%Y%m%d')
    except:
        changeDate = changeDate.replace('-', '')
    requestString.append("&changeDate=%s"%(changeDate))
    try:
        annoDate = annoDate.strftime('%Y%m%d')
    except:
        annoDate = annoDate.replace('-', '')
    requestString.append("&annoDate=%s"%(annoDate))
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
            api_base.handle_error(csvString, 'HKEqudtJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKEqudtJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'changeDate', u'equCountryCd', u'divNum', u'annoDate', u'chgReason', u'chgReasonCd', u'equTypeCd', u'pshTypeCd', u'parCurrencyCd', u'parValue', u'authCap', u'authShares', u'paidCap', u'paidShares']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'equCountryCd': 'str','chgReason': 'str','chgReasonCd': 'str','equTypeCd': 'str','pshTypeCd': 'str','parCurrencyCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKDequJLGet(partyId, annoDate = "", SHName = "", field = "", pandas = "1"):
    """
    香港公司董事持股权益信息，记录香港上市公司董事持股情况信息，包括截止日期、股东名称、权益类型、权益金额等内容
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param annoDate: 公告日期，输入格式“YYYYMMDD”,可空
    :param SHName: 董事名称，如‘李嘉诚’,可空
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
    requestString.append('/api/HKsec/getHKDequJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        annoDate = annoDate.strftime('%Y%m%d')
    except:
        annoDate = annoDate.replace('-', '')
    requestString.append("&annoDate=%s"%(annoDate))
    if not isinstance(SHName, str) and not isinstance(SHName, unicode):
        SHName = str(SHName)

    requestString.append("&SHName=%s"%(SHName))
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
            api_base.handle_error(csvString, 'HKDequJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKDequJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'endDate', u'id', u'annoDate', u'infoSourceCd', u'shName', u'equObjectCd', u'equTypeCd', u'equCharacterCd', u'identityTypeCd', u'relCompany', u'interestsNum', u'interestsAmt', u'intCurrencyCd', u'holdPct', u'calcuBenchmarkCd', u'remark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'infoSourceCd': 'str','shName': 'str','equObjectCd': 'str','equTypeCd': 'str','equCharacterCd': 'str','identityTypeCd': 'str','relCompany': 'str','intCurrencyCd': 'str','calcuBenchmarkCd': 'str','remark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKShequJLGet(partyId, annoDate = "", shName = "", field = "", pandas = "1"):
    """
    香港公司主要股东持股情况信息，记录香港上市公司主要持股人在各报告期以各种不同身份分别持有本公司的各种权益的明细信息。包括截止日期、股东名称、权益类型、权益金额等内容
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param annoDate: 公告日期，输入格式“YYYYMMDD”,可空
    :param shName: 股东名称，如‘李嘉诚’,可空
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
    requestString.append('/api/HKsec/getHKShequJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        annoDate = annoDate.strftime('%Y%m%d')
    except:
        annoDate = annoDate.replace('-', '')
    requestString.append("&annoDate=%s"%(annoDate))
    if not isinstance(shName, str) and not isinstance(shName, unicode):
        shName = str(shName)

    requestString.append("&shName=%s"%(shName))
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
            api_base.handle_error(csvString, 'HKShequJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKShequJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'endDate', u'id', u'annoDate', u'infoSourceCd', u'shName', u'equObjectCd', u'equTypeCd', u'equCharacterCd', u'identityTypeCd', u'interestsNum', u'interestsAmt', u'intCurrencyCd', u'holdPct', u'calcuBenchmarkCd', u'remark']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'infoSourceCd': 'str','shName': 'str','equObjectCd': 'str','equTypeCd': 'str','equCharacterCd': 'str','identityTypeCd': 'str','intCurrencyCd': 'str','calcuBenchmarkCd': 'str','remark': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKDequchJLGet(secId = "", partyId = "", eventDate = "", shName = "", field = "", pandas = "1"):
    """
    香港董事持股人权益变动表，记录香港上市公司董事持股变动情况信息，包括事件日期、董事知悉相关事件日期、关联公司信息、债券证信息，先前占比后目前股份及占比信息等内容
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表,secId、partyId至少选择一个
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到,secId、partyId至少选择一个
    :param eventDate: 事件日期，输入格式“YYYYMMDD”,可空
    :param shName: 股东名称，如‘李嘉诚’,可空
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
    requestString.append('/api/HKsec/getHKDequchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("&partyId=%s"%(partyId))
    try:
        eventDate = eventDate.strftime('%Y%m%d')
    except:
        eventDate = eventDate.replace('-', '')
    requestString.append("&eventDate=%s"%(eventDate))
    if not isinstance(shName, str) and not isinstance(shName, unicode):
        shName = str(shName)

    requestString.append("&shName=%s"%(shName))
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
            api_base.handle_error(csvString, 'HKDequchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKDequchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'secId', u'id', u'eventDate', u'noticeToDirDate', u'equType', u'shName', u'relCompany', u'bondCertTypeCd', u'bondCertCurrCd', u'bondCertPrvValue', u'relStkNum', u'dislTradReasonCd', u'equCharacterCd', u'sharesBfChange', u'ratioInTotBfChg', u'sharesAfChange', u'ratioInTotAfChg']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','equType': 'str','shName': 'str','relCompany': 'str','bondCertTypeCd': 'str','bondCertCurrCd': 'str','dislTradReasonCd': 'str','equCharacterCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKShequchJLGet(secId = "", partyId = "", eventDate = "", shName = "", field = "", pandas = "1"):
    """
    香港主要持股人权益变动信息，记录香港上市公司主要持股人持股变动情况信息，包括事件日期、主要股东知悉相关事件日期、股东信息，权益性质，设计股份，先前股份占比，目前股份占比
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表,secId、partyId至少选择一个
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到,secId、partyId至少选择一个
    :param eventDate: 事件日期，输入格式“YYYYMMDD”,可空
    :param shName: 股东名称，如‘李嘉诚’,可空
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
    requestString.append('/api/HKsec/getHKShequchJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("&partyId=%s"%(partyId))
    try:
        eventDate = eventDate.strftime('%Y%m%d')
    except:
        eventDate = eventDate.replace('-', '')
    requestString.append("&eventDate=%s"%(eventDate))
    if not isinstance(shName, str) and not isinstance(shName, unicode):
        shName = str(shName)

    requestString.append("&shName=%s"%(shName))
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
            api_base.handle_error(csvString, 'HKShequchJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKShequchJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'secId', u'id', u'eventDate', u'noticeToMshDate', u'equType', u'shName', u'shCharacterCd', u'dislTradeReasonCd', u'equCharacterCd', u'relStkNum', u'sharesBfChange', u'sharesAfChange', u'ratioInTotBfChg']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','equType': 'str','shName': 'str','shCharacterCd': 'str','dislTradeReasonCd': 'str','equCharacterCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFinDJLGet(partyId, sprPublishDate = "", antOrAcCd = "", field = "", pandas = "1"):
    """
    香港公司融资相关日程信息，详细记录股份发行预期和实际的相关日期
    
    :param partyId: 机构ID，法人统一编码，可通过法人名称在DataAPI.PartyIDGet获取到
    :param sprPublishDate: 公布发售价日期，输入格式“YYYYMMDD”,可空
    :param antOrAcCd: 预期或实际参数编码。对应DataAPI.SysCodeGet.codeTypeID=50034。,可空
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
    requestString.append('/api/HKsec/getHKFinDJL.csv?ispandas=1&') 
    if not isinstance(partyId, str) and not isinstance(partyId, unicode):
        partyId = str(partyId)

    requestString.append("partyId=%s"%(partyId))
    try:
        sprPublishDate = sprPublishDate.strftime('%Y%m%d')
    except:
        sprPublishDate = sprPublishDate.replace('-', '')
    requestString.append("&sprPublishDate=%s"%(sprPublishDate))
    if not isinstance(antOrAcCd, str) and not isinstance(antOrAcCd, unicode):
        antOrAcCd = str(antOrAcCd)

    requestString.append("&antOrAcCd=%s"%(antOrAcCd))
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
            api_base.handle_error(csvString, 'HKFinDJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFinDJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'partyId', u'fcgNum', u'relationFlag', u'antOrAcCd', u'priceDate', u'sprPublishDate', u'regSubsBegin', u'regSubsEnd', u'elecSubsEndDate', u'subsWaplFormEnd', u'subYaplFormEnd', u'subBlaplFormEnd', u'subPaplFormEnd', u'rightLatradeDate', u'exDate', u'recordDate', u'recdDate', u'sendPospDocuDate', u'payEndDate', u'underwagUncodDate', u'regBegin', u'regEnd', u'issueResPubDate', u'shaRelseDate', u'refundOutDate', u'listDate', u'oallotListDate', u'oaoExpDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'antOrAcCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKSnotJLGet(secId, publishDate = "", field = "", pandas = "1"):
    """
    香港证券特别提示表，记录各证券品种的交易提示信息，例如：停牌提示、会议提示、分红提示、分拆\合并提示、简称变更提示等
    
    :param secId: 证券ID，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param publishDate: 公布日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/HKsec/getHKSnotJL.csv?ispandas=1&') 
    requestString.append("secId=")
    if hasattr(secId,'__iter__') and not isinstance(secId, str):
        if len(secId) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = secId
            requestString.append(None)
        else:
            requestString.append(','.join(secId))
    else:
        requestString.append(secId)
    try:
        publishDate = publishDate.strftime('%Y%m%d')
    except:
        publishDate = publishDate.replace('-', '')
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
            api_base.handle_error(csvString, 'HKSnotJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKSnotJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secId', u'noticeDate', u'id', u'noticeTypeCd', u'publishDate', u'noticeCont', u'haltBeginTime', u'resumpBeginTime', u'haltPeriod', u'haltTypeCd', u'haltReasonTypeCd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secId': 'str','noticeTypeCd': 'str','noticeCont': 'str','haltPeriod': 'str','haltTypeCd': 'str','haltReasonTypeCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def IdxHkJLGet(secID = "", ticker = "", pubOrgCD = "", listStatusCD = "", field = "", pandas = "1"):
    """
    获取港股指数的基本要素信息，包括指数名称、指数代码、指数类型、发布机构、编制机构、发布日期、基日、基点、加权方式等。
    
    :param secID: 指数展示代码,如:HSI.ZIHK 恒生指数,可以是列表,secID、ticker、pubOrgCD至少选择一个
    :param ticker: 指数代码,如:HSI 恒生指数,可以是列表,secID、ticker、pubOrgCD至少选择一个
    :param pubOrgCD: 发布机构编码,如:31576 恒生指数有限公司,可以是列表,secID、ticker、pubOrgCD至少选择一个
    :param listStatusCD: 指数当前状态,可以是列表,可空
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
    requestString.append('/api/HKsec/getIdxHkJL.csv?ispandas=1&') 
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
    requestString.append("&pubOrgCD=")
    if hasattr(pubOrgCD,'__iter__') and not isinstance(pubOrgCD, str):
        if len(pubOrgCD) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = pubOrgCD
            requestString.append(None)
        else:
            requestString.append(','.join(pubOrgCD))
    else:
        requestString.append(pubOrgCD)
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
            api_base.handle_error(csvString, 'IdxHkJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'IdxHkJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secFullName', u'secFullNameEn', u'secShortName', u'secShortNameEn', u'ticker', u'exchange', u'pubOrgCD', u'porgFullName', u'porgFullNameEn', u'creatOrgCD', u'corgFullName', u'corgFullNameEn', u'publishDate', u'bCalcDate', u'baseDate', u'basePoint', u'consNum', u'indexTypeCD', u'consTypeName', u'indexPrepObj', u'corgIndexType', u'wMethod', u'publishFreq', u'currencyCD', u'ifAccuDiv', u'consAdPeriod', u'indexAdPeriod', u'listStatusCD', u'indexDesc', u'consSamSel', u'consSelMethod', u'indCalcMethod', u'addDesc', u'isMkt']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secFullName': 'str','secFullNameEn': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','exchange': 'str','porgFullName': 'str','porgFullNameEn': 'str','corgFullName': 'str','corgFullNameEn': 'str','indexTypeCD': 'str','consTypeName': 'str','indexPrepObj': 'str','corgIndexType': 'str','wMethod': 'str','currencyCD': 'str','ifAccuDiv': 'str','consAdPeriod': 'str','indexAdPeriod': 'str','listStatusCD': 'str','indexDesc': 'str','consSamSel': 'str','consSelMethod': 'str','indCalcMethod': 'str','addDesc': 'str','isMkt': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def IdxConsHkJLGet(secID = "", ticker = "", consID = "", consticker = "", isNew = "", field = "", pandas = "1"):
    """
    获取港股市场上主要指数的成分构成情况，包括指数成分证券名称、成分证券代码、入选日期、剔除日期等。
    
    :param secID: 指数展示代码,如:HSI.ZIHK 恒生指数,可以是列表,secID、ticker至少选择一个
    :param ticker: 指数代码,如:HSI 恒生指数,可以是列表,secID、ticker至少选择一个
    :param consID: 成分证券展示代码,如:601318.XSHG 中国平安保险(集团)股份有限公司,可以是列表,可空
    :param consticker: 成分证券代码,如:601318 中国平安保险(集团)股份有限公司,可以是列表,可空
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
    requestString.append('/api/HKsec/getIdxConsHkJL.csv?ispandas=1&') 
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
    requestString.append("&consID=")
    if hasattr(consID,'__iter__') and not isinstance(consID, str):
        if len(consID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = consID
            requestString.append(None)
        else:
            requestString.append(','.join(consID))
    else:
        requestString.append(consID)
    requestString.append("&consticker=")
    if hasattr(consticker,'__iter__') and not isinstance(consticker, str):
        if len(consticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = consticker
            requestString.append(None)
        else:
            requestString.append(','.join(consticker))
    else:
        requestString.append(consticker)
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
            api_base.handle_error(csvString, 'IdxConsHkJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'IdxConsHkJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'secShortName', u'secShortNameEn', u'ticker', u'consID', u'consShortName', u'consticker', u'consExchangeCD', u'intoDate', u'outDate', u'isNew', u'addDesc']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','secShortName': 'str','secShortNameEn': 'str','ticker': 'str','consID': 'str','consShortName': 'str','consticker': 'str','consExchangeCD': 'str','isNew': 'str','addDesc': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def BondInfoJLGet(secID = "", ticker = "", typeID = "", field = "", pandas = "1"):
    """
    1.本表记录债券基本信息，囊括所有债券基本信息，包括名称、类型、发行人等内容；2.历史数据追溯至1987年，每日更新。
    
    :param secID: 证券ID,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param typeID: 债券分类ID。例如，0202010101-国债；0202010201-央行票据。对应DataAPI.SysCodeGet.codeTypeID=30018。,可空
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
    requestString.append('/api/bond/getBondInfoJL.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'BondInfoJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'BondInfoJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'bondID', u'secID', u'ticker', u'exchangeCD', u'secShortName', u'secFullName', u'bondForm', u'typeID', u'partyID', u'issuer', u'issuerType', u'currencyCD', u'listDate', u'listStatus', u'delistDate', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'bondID': 'str','secID': 'str','ticker': 'str','exchangeCD': 'str','secShortName': 'str','secFullName': 'str','bondForm': 'str','typeID': 'str','issuer': 'str','issuerType': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __MktEquwJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表为衍生表，主要计算个股每周的市场表现。包括最高价、最低价、涨跌、涨跌幅、均价、对数收益率、BETA值等，历史追溯至1990年，每周更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'000001'，可获取到ID'000001.XSHE'后，在此输入'000001.XSHE',可以是列表
    :param startDate: 起始日期，输入格式为yyyymmdd,可空
    :param finishDate: 结束日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/market/getMktEquwJL.csv?ispandas=1&') 
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
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, '__MktEquwJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__MktEquwJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'endDate', u'firstTradeDate', u'lastTradeDate', u'numDays', u'preClosePrice', u'openPrice', u'highestPrice', u'dayHigh', u'lowestPrice', u'dayLow', u'closePrice', u'highClosePrice', u'dayHcp', u'lowClosePrice', u'dayLcp', u'avgPrice', u'rangePct', u'adChg', u'adChgPct', u'logReturn', u'turnoverVol', u'turnoverValue', u'turnoverRate', u'avgTurnoverRate', u'adPreClosePrice', u'adOpenPrice', u'adHighestPrice', u'adDayHigh', u'adDayLow', u'adLowestPrice', u'adClosePrice', u'adHighClosePrice', u'adLowClosePrice', u'adDayHcp', u'adDayLcp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktEquPEJLGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    本表主要记录中证指数公司发布的沪深两市A股市盈率，包含发布机构代码,市盈率,滚动市盈率,市净率,证监会行业代码等，历史数据追溯至2011年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'000001'，可获取到ID'000001.XSHE'后，在此输入'000001.XSHE'
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
    requestString.append('/api/market/getMktEquPEJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
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
            api_base.handle_error(csvString, 'MktEquPEJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktEquPEJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'infoSource', u'PE', u'PETTM', u'PB', u'sacIndustryCD']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str','infoSource': 'str','sacIndustryCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktEqudJLGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    本表记录股票交易日行情，包括昨收、开盘价、收盘价、最高、最低价、成交价、成交量、成成交笔数、5档申报买入价、量等内容，历史追溯至1990年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'000001'，可获取到ID'000001.XSHE'后，在此输入'000001.XSHE'
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
    requestString.append('/api/market/getMktEqudJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
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
            api_base.handle_error(csvString, 'MktEqudJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktEqudJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'turnovervalue', u'turnoverVol', u'dealAmount', u'chg', u'chgPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktEqumJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表为衍生表，主要计算个股每月的市场表现。包括开盘价、收盘价、涨跌、涨跌幅、均价、对数收益率、年化波动率、年化收益率、BETA值等。历史追溯至1990年，每月更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'000001'，可获取到ID'000001.XSHE'后，在此输入'000001.XSHE',可以是列表
    :param startDate: 起始日期，输入格式为yyyymmdd(每月月末),可空
    :param finishDate: 结束日期，输入格式为yyyymmdd(每月月末),可空
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
    requestString.append('/api/market/getMktEqumJL.csv?ispandas=1&') 
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
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, 'MktEqumJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktEqumJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'endDate', u'firstTradeDate', u'lastTradeDate', u'numDays', u'preClosePrice', u'openPrice', u'highestPrice', u'dayHigh', u'lowestPrice', u'dayLow', u'closePrice', u'highClosePrice', u'dayHcp', u'lowClosePrice', u'dayLcp', u'avgPrice', u'rangePct', u'adChg', u'adChgPct', u'logReturn', u'turnoverVol', u'turnoverValue', u'turnoverRate', u'avgTurnoverRate', u'volat24m', u'volat60m', u'beta24m', u'beta60m', u'yield24m', u'yield60m', u'adPreClosePrice', u'adOpenPrice', u'adHighestPrice', u'adDayHigh', u'adDayLow', u'adLowestPrice', u'adClosePrice', u'adHighClosePrice', u'adLowClosePrice', u'adDayHcp', u'adDayLcp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktEquyJLGet(secID, beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    本表为衍生表，主要计算个股每年的市场表现。包括收盘价、均价、涨跌、涨跌幅、交易天数、平均换手率等，历史追溯至1990年。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'E',输入'000001'，可获取到ID'000001.XSHE'后，在此输入'000001.XSHE'
    :param beginDate: 起始日期，输入格式为yyyymmdd,可空
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
    requestString.append('/api/market/getMktEquyJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
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
            api_base.handle_error(csvString, 'MktEquyJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktEquyJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'endDate', u'firstTradeDate', u'lastTradeDate', u'numDays', u'preClosePrice', u'openPrice', u'highestPrice', u'dayHigh', u'lowestPrice', u'dayLow', u'closePrice', u'highClosePrice', u'dayHcp', u'lowClosePrice', u'dayLcp', u'avgPrice', u'rangePct', u'adChg', u'adChgPct', u'logReturn', u'turnoverVol', u'turnoverValue', u'turnoverRate', u'avgTurnoverRate', u'adPreClosePrice', u'adOpenPrice', u'adHighestPrice', u'adDayHigh', u'adDayLow', u'adLowestPrice', u'adClosePrice', u'adHighClosePrice', u'adLowClosePrice', u'adDayHcp', u'adDayLcp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktFunddJLGet(secID, startDate = "", endDate = "", field = "", pandas = "1"):
    """
    本表主要记录沪深两市基金日行情，包括昨收盘,开盘价,最高价,最低价,收盘价,成交金额,成交量,成交笔数,涨跌,涨跌幅等，历史追溯至1998年，每日更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'F',输入'510050'，可获取到ID'510050.XSHG'后，在此输入'510050.XSHG'
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
    requestString.append('/api/market/getMktFunddJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
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
            api_base.handle_error(csvString, 'MktFunddJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktFunddJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'tradeDate', u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', u'turnovervalue', u'turnoverVol', u'dealAmount', u'chg', u'chgPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktFundmJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表主要记录沪深两市基金月行情，包括首个交易日,最后交易日,交易天数,昨收价,开盘价,最高价,最高价日,最低价,最低价日,收盘价,最高收盘价,最高收盘价日,最低收盘价,最低收盘价日,均价,振幅等，历史追溯至1993年，每月更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'F',输入'510050'，可获取到ID'510050.XSHG'后，在此输入'510050.XSHG'
    :param startDate: 起始日期，输入格式为yyyymmdd（每月月末）,可空
    :param finishDate: 结束日期，输入格式为yyyymmdd（每月月末）,可空
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
    requestString.append('/api/market/getMktFundmJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, 'MktFundmJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktFundmJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'endDate', u'firstTradeDate', u'lastTradeDate', u'numDays', u'preClosePrice', u'openPrice', u'highestPrice', u'dayHigh', u'lowestPrice', u'dayLow', u'closePrice', u'highClosePrice', u'dayHcp', u'lowClosePrice', u'dayLcp', u'avgPrice', u'rangePct', u'Chg', u'ChgPct', u'turnoverVol', u'turnoverValue', u'turnoverRate', u'avgTurnoverRate', u'avgDiscount', u'avgDiscountRatio', u'adPreClosePrice', u'adOpenPrice', u'adHighestPrice', u'adDayHigh', u'adDayLow', u'adLowestPrice', u'adClosePrice', u'adHighClosePrice', u'adLowClosePrice', u'adDayHcp', u'adDayLcp', u'relChgPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktFundwJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表主要记录沪深两市基金周行情，包括首个交易日,最后交易日,交易天数,昨收价,开盘价,最高价,最高价日,最低价,最低价日,收盘价,最高收盘价,最高收盘价日,最低收盘价,最低收盘价日,均价,振幅等，历史追溯至1993年，每周更新。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'F',输入'510050'，可获取到ID'510050.XSHG'后，在此输入'510050.XSHG'
    :param startDate: 起始日期，输入格式为yyyymmdd（每周最后一个交易日）,可空
    :param finishDate: 结束日期，输入格式为yyyymmdd（每周最后一个交易日）,可空
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
    requestString.append('/api/market/getMktFundwJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, 'MktFundwJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktFundwJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'endDate', u'firstTradeDate', u'lastTradeDate', u'numDays', u'preClosePrice', u'openPrice', u'highestPrice', u'dayHigh', u'lowestPrice', u'dayLow', u'closePrice', u'highClosePrice', u'dayHcp', u'lowClosePrice', u'dayLcp', u'avgPrice', u'rangePct', u'Chg', u'ChgPct', u'turnoverVol', u'turnoverValue', u'turnoverRate', u'avgTurnoverRate', u'avgDiscount', u'avgDiscountRatio', u'adPreClosePrice', u'adOpenPrice', u'adHighestPrice', u'adDayHigh', u'adDayLow', u'adLowestPrice', u'adClosePrice', u'adHighClosePrice', u'adLowClosePrice', u'adDayHcp', u'adDayLcp', u'relChgPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def MktFundyJLGet(secID, startDate = "", finishDate = "", field = "", pandas = "1"):
    """
    本表主要记录沪深两市基金年行情，包括首个交易日,最后交易日,交易天数,昨收价,开盘价,最高价,最高价日,最低价,最低价日,收盘价,最高收盘价,最高收盘价日,最低收盘价,最低收盘价日,均价,振幅等，历史追溯至1993年。
    
    :param secID: 证券内部编码，一串流水号,可先通过DataAPI.SecIDGet获取到，如在DataAPI.SecIDGet，选择证券类型为'F',输入'510050'，可获取到ID'510050.XSHG'后，在此输入'510050.XSHG'
    :param startDate: 起始日期，输入格式为yyyymmdd（每周最后一个交易日）,可空
    :param finishDate: 结束日期，输入格式为yyyymmdd（每周最后一个交易日）,可空
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
    requestString.append('/api/market/getMktFundyJL.csv?ispandas=1&') 
    if not isinstance(secID, str) and not isinstance(secID, unicode):
        secID = str(secID)

    requestString.append("secID=%s"%(secID))
    try:
        startDate = startDate.strftime('%Y%m%d')
    except:
        startDate = startDate.replace('-', '')
    requestString.append("&startDate=%s"%(startDate))
    try:
        finishDate = finishDate.strftime('%Y%m%d')
    except:
        finishDate = finishDate.replace('-', '')
    requestString.append("&finishDate=%s"%(finishDate))
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
            api_base.handle_error(csvString, 'MktFundyJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'MktFundyJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'exchangeCD', u'endDate', u'firstTradeDate', u'lastTradeDate', u'numDays', u'preClosePrice', u'openPrice', u'highestPrice', u'dayHigh', u'lowestPrice', u'dayLow', u'closePrice', u'highClosePrice', u'dayHcp', u'lowClosePrice', u'dayLcp', u'avgPrice', u'rangePct', u'Chg', u'ChgPct', u'turnoverVol', u'turnoverValue', u'turnoverRate', u'avgTurnoverRate', u'avgDiscount', u'avgDiscountRatio', u'adPreClosePrice', u'adOpenPrice', u'adHighestPrice', u'adDayHigh', u'adDayLow', u'adLowestPrice', u'adClosePrice', u'adHighClosePrice', u'adLowClosePrice', u'adDayHcp', u'adDayLcp', u'relChgPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','exchangeCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def ReportJLGet(secID = "", ticker = "", orgName = "", BeginWriteDate = "", EndwriteDate = "", title = "", author = "", field = "", pandas = "1"):
    """
    巨灵数据库的研报基本信息以及预测数据信息。历史数据从2008年开始，涵盖A股及港股市场，数据包含EPS、PE、营业收入、净利润、净资产收益率、市净率、评级、目标价等内容。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param orgName: 研究机构名称，通过输入机构名称，获得该机构发布的研究报告,可以是列表,可空
    :param BeginWriteDate: 研报书写日期，书写起始日期，获得自该日期开始撰写的研究报告，输入格式“YYYYMMDD”,可空
    :param EndwriteDate: 研报书写日期，可获得到某日期之前撰写的研报数据，输入格式“YYYYMMDD”,可空
    :param title: 研报标题，可输入个别关键词，通过模糊匹配，获得相关研报,可空
    :param author: 研报作者名称，可输入多个,可以是列表,可空
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
    requestString.append('/api/RRP/getReportJL.csv?ispandas=1&') 
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
    requestString.append("&orgName=")
    if hasattr(orgName,'__iter__') and not isinstance(orgName, str):
        if len(orgName) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = orgName
            requestString.append(None)
        else:
            requestString.append(','.join(orgName))
    else:
        requestString.append(orgName)
    try:
        BeginWriteDate = BeginWriteDate.strftime('%Y%m%d')
    except:
        BeginWriteDate = BeginWriteDate.replace('-', '')
    requestString.append("&BeginWriteDate=%s"%(BeginWriteDate))
    try:
        EndwriteDate = EndwriteDate.strftime('%Y%m%d')
    except:
        EndwriteDate = EndwriteDate.replace('-', '')
    requestString.append("&EndwriteDate=%s"%(EndwriteDate))
    if not isinstance(title, str) and not isinstance(title, unicode):
        title = str(title)

    requestString.append("&title=%s"%(title))
    requestString.append("&author=")
    if hasattr(author,'__iter__') and not isinstance(author, str):
        if len(author) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = author
            requestString.append(None)
        else:
            requestString.append(','.join(author))
    else:
        requestString.append(author)
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
            api_base.handle_error(csvString, 'ReportJLGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'ReportJLGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'orgName', u'writeDate', u'secID', u'ticker', u'secShortName', u'title', u'author', u'ForecastPeriodTime', u'EPS', u'PE', u'sales', u'netProfit', u'PnetProfit', u'ROE', u'PB', u'recommdation', u'adjustRecommdation', u'ratingChange', u'targetPrice', u'targetPriceStarteDate', u'targetPriceExpireDate', u'publishDate', u'intoDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'orgName': 'str','writeDate': 'str','secID': 'str','ticker': 'str','secShortName': 'str','title': 'str','author': 'str','ForecastPeriodTime': 'str','recommdation': 'str','adjustRecommdation': 'str','ratingChange': 'str','targetPrice': 'str','targetPriceStarteDate': 'str','targetPriceExpireDate': 'str','publishDate': 'str','intoDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

