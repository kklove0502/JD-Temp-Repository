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

__doc__="TEJ"
def HKEquTEJGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    香港证券基础资料，涵盖交易代码，交易板块，所属行业，ISIN码，对应A股B股代码，电话传真网址，高管，财务总监，董事会秘书信息，注册地址，公司及集团名，承销价格，每股面值，公司设立时间，首次上次时间，借壳或转板上市时间，退市时间，服务代理单位及其电话和地址，证券名称更名历史记录，证券交易单位更改的历史记录
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/party/getHKEquTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKEquTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKEquTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exchange', u'hsIndCls', u'hsSector', u'TEJSector', u'finType', u'ISIN', u'tickerA', u'tickerB', u'tel', u'fax', u'website', u'manager1', u'manager2', u'manager3', u'manager4', u'CFO', u'boardSect', u'regPlace', u'secFullName', u'secFUllNameEn', u'copAdd', u'grpName', u'subPrice', u'parValue', u'parValueCurr', u'tradingUnit', u'endDateAccYear', u'mainBusiness', u'mainproPro', u'empSum', u'estbDate', u'IPODate', u'backdoorDate', u'listDate', u'delistDate', u'svcAgent', u'svcTel', u'svcAdd', u'chgName1', u'chgNameEn1', u'chgDate1', u'chgName2', u'chgNameEn2', u'chgDate2', u'chgName3', u'chgNameEn3', u'chgDate3', u'chgName4', u'chgNameEn4', u'chgDate4', u'chgName5', u'chgNameEn5', u'chgDate5', u'chgName6', u'chgNameEn6', u'chgDate6', u'chgName7', u'chgNameEn7', u'chgDate7', u'chgName8', u'chgNameEn8', u'chgDate8', u'chgName9', u'chgNameEn9', u'chgDate9', u'chgName10', u'chgNameEn10', u'chgDate10', u'chgTik1', u'chgTikDate1', u'chgTik2', u'chgTikDate2', u'chgTik3', u'chgTikDate3', u'chgTik4', u'chgTikDate4']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exchange': 'str','hsIndCls': 'str','hsSector': 'str','TEJSector': 'str','finType': 'str','ISIN': 'str','tickerA': 'str','tickerB': 'str','tel': 'str','fax': 'str','website': 'str','manager1': 'str','manager2': 'str','manager3': 'str','manager4': 'str','CFO': 'str','boardSect': 'str','regPlace': 'str','secFullName': 'str','secFUllNameEn': 'str','copAdd': 'str','grpName': 'str','parValueCurr': 'str','endDateAccYear': 'str','mainBusiness': 'str','mainproPro': 'str','estbDate': 'str','IPODate': 'str','backdoorDate': 'str','listDate': 'str','delistDate': 'str','svcAgent': 'str','svcTel': 'str','svcAdd': 'str','chgName1': 'str','chgNameEn1': 'str','chgDate1': 'str','chgName2': 'str','chgNameEn2': 'str','chgDate2': 'str','chgName3': 'str','chgNameEn3': 'str','chgDate3': 'str','chgName4': 'str','chgNameEn4': 'str','chgDate4': 'str','chgName5': 'str','chgNameEn5': 'str','chgDate5': 'str','chgName6': 'str','chgNameEn6': 'str','chgDate6': 'str','chgName7': 'str','chgNameEn7': 'str','chgDate7': 'str','chgName8': 'str','chgNameEn8': 'str','chgDate8': 'str','chgName9': 'str','chgNameEn9': 'str','chgDate9': 'str','chgName10': 'str','chgNameEn10': 'str','chgDate10': 'str','chgTik1': 'str','chgTikDate1': 'str','chgTik2': 'str','chgTikDate2': 'str','chgTik3': 'str','chgTikDate3': 'str','chgTik4': 'str','chgTikDate4': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKdivTEJGet(secID = "", ticker = "", annoDate = "", field = "", pandas = "1"):
    """
    香港现金股息资料，涵盖分配年度开始日期，分配年度结束日期，以原币计算的现金股息合计，一般股息和特别股息，股息币别，以港币计算的现金股息，方案公布日，除息日，股息发放日，停止过户开始日期，停止过户结束日期，最后过户日期，是否以股代息
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param annoDate: 港股分红派息公告日期,如'20150101',可空
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
    requestString.append('/api/party/getHKdivTEJ.csv?ispandas=1&') 
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
    if not isinstance(annoDate, str) and not isinstance(annoDate, unicode):
        annoDate = str(annoDate)

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
            api_base.handle_error(csvString, 'HKdivTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKdivTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exDate', u'totalCashDiv', u'genDiv', u'speDiv', u'currencyCd', u'attrPerBegin', u'attrPerEnd', u'cashDivHKD', u'payDate', u'scrip', u'annoDate', u'nonRegBegin', u'nonRegEnd', u'recordDate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','currencyCd': 'str','attrPerBegin': 'str','attrPerEnd': 'str','payDate': 'str','scrip': 'str','annoDate': 'str','nonRegBegin': 'str','nonRegEnd': 'str','recordDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKbsTEJGet(secID = "", ticker = "", endAccPeriod = "", shName = "", field = "", pandas = "1"):
    """
    香港董事持股资料，涵盖披露报告月份，持股人姓名，职称，目前持股，直接权益，间接权益，家族权益，公司权益，其他权益，持股率
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param endAccPeriod: 港股会计报告披露月份如'201506',可空
    :param shName: 持股人或者机构名称,如'甘庆林',可空
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
    requestString.append('/api/party/getHKbsTEJ.csv?ispandas=1&') 
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
    if not isinstance(endAccPeriod, str) and not isinstance(endAccPeriod, unicode):
        endAccPeriod = str(endAccPeriod)

    requestString.append("&endAccPeriod=%s"%(endAccPeriod))
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
            api_base.handle_error(csvString, 'HKbsTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKbsTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'endAccPeriod', u'shName', u'title1', u'title2', u'title3', u'curInt', u'dirInt', u'indirInt', u'famInt', u'comInt', u'otherInt', u'holdPct']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','endAccPeriod': 'str','shName': 'str','title1': 'str','title2': 'str','title3': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKinvTEJGet(secID = "", ticker = "", endAccPeriod = "", investTarget = "", field = "", pandas = "1"):
    """
    香港上市公司对外长期投资明细，涵盖会计年度结束月份，投资对象名称，持股率，注册地，发行股本币别，发行股本，主要业务
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param endAccPeriod: 会计年度结束月份,如'201412',可空
    :param investTarget: 投资公司名称,如'保特威投资有限公司',可空
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
    requestString.append('/api/party/getHKinvTEJ.csv?ispandas=1&') 
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
    if not isinstance(endAccPeriod, str) and not isinstance(endAccPeriod, unicode):
        endAccPeriod = str(endAccPeriod)

    requestString.append("&endAccPeriod=%s"%(endAccPeriod))
    if not isinstance(investTarget, str) and not isinstance(investTarget, unicode):
        investTarget = str(investTarget)

    requestString.append("&investTarget=%s"%(investTarget))
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
            api_base.handle_error(csvString, 'HKinvTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKinvTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'endAccPeriod', u'investTarget', u'holdPct', u'regCountryCd', u'issCurrCd', u'issAmt', u'mainBusiness']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','endAccPeriod': 'str','investTarget': 'str','regCountryCd': 'str','issCurrCd': 'str','mainBusiness': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKIPOTEJGet(secID = "", ticker = "", field = "", pandas = "1"):
    """
    香港上市公司IPO资料，涵盖股票IPO的承销类别，承销股数，承销价格，承销价格币别，承销公告日，公开承销开始日期，公开承销结束日期，承销抽签日期，中签公告日期，承销缴款开始日期，承销缴款结束日期，新股发放日期，主承销上和协办承销商信息，上市日期，首日收盘价，收盘价币别，上市总市值
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/party/getHKIPOTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKIPOTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKIPOTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'date', u'undCat', u'undCatEn', u'undShares', u'undPrice', u'undPriceCurr', u'annoDate', u'undBegin', u'undEnd', u'drawDate', u'drawAnnoDate', u'payDateBegin', u'payDateEnd', u'shaRelseDate', u'udwhtLead', u'udwht1', u'udwht2', u'udwht3', u'udwht4', u'udwht5', u'udwht6', u'udwht7', u'udwht8', u'udwht9', u'udwht10', u'udwht11', u'udwht12', u'udwht13', u'udwht14', u'udwht15', u'udwht16', u'udwht17', u'udwht18', u'udwht19', u'udwht20', u'udwht21', u'udwht22', u'udwht23', u'udwht24', u'udwht25', u'udwht26', u'udwht27', u'udwht28', u'udwht29', u'exchange', u'listDate', u'firstDayClosePrice', u'cloPriCurrCD', u'markerValue']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','date': 'str','undCat': 'str','undCatEn': 'str','undPriceCurr': 'str','annoDate': 'str','undBegin': 'str','undEnd': 'str','drawDate': 'str','drawAnnoDate': 'str','payDateBegin': 'str','payDateEnd': 'str','shaRelseDate': 'str','udwhtLead': 'str','udwht1': 'str','udwht2': 'str','udwht3': 'str','udwht4': 'str','udwht5': 'str','udwht6': 'str','udwht7': 'str','udwht8': 'str','udwht9': 'str','udwht10': 'str','udwht11': 'str','udwht12': 'str','udwht13': 'str','udwht14': 'str','udwht15': 'str','udwht16': 'str','udwht17': 'str','udwht18': 'str','udwht19': 'str','udwht20': 'str','udwht21': 'str','udwht22': 'str','udwht23': 'str','udwht24': 'str','udwht25': 'str','udwht26': 'str','udwht27': 'str','udwht28': 'str','udwht29': 'str','exchange': 'str','listDate': 'str','cloPriCurrCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKmpTEJGet(secID = "", ticker = "", endAccPeriod = "", field = "", pandas = "1"):
    """
    香港上市公司主营产品构成资料，涵盖披露报告月份，营业项目，该项目的收入，收入占公司总收入的比率，营业收入的币别
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param endAccPeriod: 会计报告披露月份,如'201506',可空
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
    requestString.append('/api/party/getHKmpTEJ.csv?ispandas=1&') 
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
    if not isinstance(endAccPeriod, str) and not isinstance(endAccPeriod, unicode):
        endAccPeriod = str(endAccPeriod)

    requestString.append("&endAccPeriod=%s"%(endAccPeriod))
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
            api_base.handle_error(csvString, 'HKmpTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKmpTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'endAccPeriod', u'bussItem', u'income', u'incRatio', u'currencyCd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','endAccPeriod': 'str','bussItem': 'str','currencyCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKsmTEJGet(secID = "", ticker = "", smDate = "", field = "", pandas = "1"):
    """
    香港上市公司股东会资料，涵盖股东会日期，股东会年度，股东会时间，董事会日期，是否常会，现金股利，盈余配股，公积配股，无偿配股合计，股息币别，股东会当日股价，现金认购百分比，认购价格，现金增资额，减资，增资前股本，最后过户日，停止过户开始日期，停止过户结束日期，开会地点
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param smDate: 股东会日期,如'20150520',可空
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
    requestString.append('/api/party/getHKsmTEJ.csv?ispandas=1&') 
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
    if not isinstance(smDate, str) and not isinstance(smDate, unicode):
        smDate = str(smDate)

    requestString.append("&smDate=%s"%(smDate))
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
            api_base.handle_error(csvString, 'HKsmTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKsmTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'smDate', u'year', u'mtTime', u'rglMt', u'boaMtDate', u'cashDiv', u'stcDivEar', u'stoDivCap', u'subTotalDiv', u'currencyCd', u'mtDatePrice', u'subsRatio', u'subsPrice', u'rigIss', u'capDecr', u'preDivCap', u'recordDate', u'nonRegBegin', u'nonRegEnd', u'mtAdd', u'mtAddEn']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','smDate': 'str','year': 'str','mtTime': 'str','rglMt': 'str','boaMtDate': 'str','currencyCd': 'str','recordDate': 'str','nonRegBegin': 'str','nonRegEnd': 'str','mtAdd': 'str','mtAddEn': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKppTEJGet(secID = "", ticker = "", annoDate = "", field = "", pandas = "1"):
    """
    香港上市公司私募发行资料，涵盖已发行股本，发行证券种类，定价日期，预期完成日期，实际完成日期，股东会决议日期，增资前股数，私募股数，配售价格，配售价币别
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param annoDate: 公告发布日期,如'20121213',可空
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
    requestString.append('/api/party/getHKppTEJ.csv?ispandas=1&') 
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
    if not isinstance(annoDate, str) and not isinstance(annoDate, unicode):
        annoDate = str(annoDate)

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
            api_base.handle_error(csvString, 'HKppTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKppTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'annoDate', u'seq', u'year', u'market', u'issueCap', u'secType', u'priceDate', u'dealDate', u'realDate', u'mtDate', u'sharesBfChance', u'share', u'price', u'currencyCd', u'sharesAfChange', u'bppRatio', u'appRatio', u'totalAmt', u'netAmt', u'finYear', u'finNetValue', u'capPrice', u'capPur', u'capPurEn', u'erase', u'eraseAnnoDate', u'agent', u'agentEn', u'fundHold1', u'prvRatio1', u'fundHold2', u'prvRatio2', u'fundHold3', u'prvRatio3', u'fundHold4', u'prvRatio4', u'fundHold5', u'prvRatio5', u'fundHold6', u'prvRatio6', u'fundHold7', u'prvRatio7', u'fundHold8', u'prvRatio8', u'fundHold9', u'prvRatio9', u'fundHold10', u'prvRatio10', u'fundHold11', u'prvRatio11', u'fundHold12', u'prvRatio12', u'fundHold13', u'prvRatio13', u'fundHold14', u'prvRatio14', u'fundHold15', u'prvRatio15', u'fundHold16', u'prvRatio16', u'fundHold17', u'prvRatio17', u'fundHold18', u'prvRatio18', u'fundHold19', u'prvRatio19', u'fundHold20', u'prvRatio20']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','annoDate': 'str','seq': 'str','year': 'str','market': 'str','secType': 'str','priceDate': 'str','dealDate': 'str','realDate': 'str','mtDate': 'str','currencyCd': 'str','finYear': 'str','finNetValue': 'str','capPur': 'str','capPurEn': 'str','erase': 'str','eraseAnnoDate': 'str','agent': 'str','agentEn': 'str','fundHold1': 'str','prvRatio1': 'str','fundHold2': 'str','prvRatio2': 'str','fundHold3': 'str','prvRatio3': 'str','fundHold4': 'str','prvRatio4': 'str','fundHold5': 'str','prvRatio5': 'str','fundHold6': 'str','prvRatio6': 'str','fundHold7': 'str','prvRatio7': 'str','fundHold8': 'str','prvRatio8': 'str','fundHold9': 'str','prvRatio9': 'str','fundHold10': 'str','prvRatio10': 'str','fundHold11': 'str','prvRatio11': 'str','fundHold12': 'str','prvRatio12': 'str','fundHold13': 'str','prvRatio13': 'str','fundHold14': 'str','prvRatio14': 'str','fundHold15': 'str','prvRatio15': 'str','fundHold16': 'str','prvRatio16': 'str','fundHold17': 'str','prvRatio17': 'str','fundHold18': 'str','prvRatio18': 'str','fundHold19': 'str','prvRatio19': 'str','fundHold20': 'str','prvRatio20': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKcaTEJGet(secID = "", ticker = "", exDate = "", field = "", pandas = "1"):
    """
    香港上市公司股份变动资料，涵盖变动前总股数，变动后总股数，现金增资股数，供股股数，以股代息股数，盈余送股股数，公积送股股数，认股权证转股股数，购股权转股股数，债券转股股数，回购注销股数，其他股数变动股数，现增认购价，认购价币别，配股认购价，配股认购价币别，合并比率，分割比率，现金增资率，盈余增资率，公积增资率，股本减资额，变动后总股本，股本币别，面额，变动公布日，停止过户起始日，停止过户结束日期，原股东缴款开始日期，新股上市日期，新股发放日期，最后过户日期，除权日期
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param exDate: 除权日,如'20150101',可空
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
    requestString.append('/api/party/getHKcaTEJ.csv?ispandas=1&') 
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
    if not isinstance(exDate, str) and not isinstance(exDate, unicode):
        exDate = str(exDate)

    requestString.append("&exDate=%s"%(exDate))
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
            api_base.handle_error(csvString, 'HKcaTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKcaTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'exDate', u'stkNo', u'newShare', u'rigIss', u'scrip', u'bonusIssEar', u'bonusIssRes', u'shareIncWts', u'shareIncOpt', u'shareIncBond', u'shareRepur', u'othChg', u'placePrice', u'placePriCurr', u'rigIssPrice', u'rigIssPriCurr', u'consRatio', u'splitRatio', u'rigIssRatio', u'bonusIssEarRatio', u'bonusIssResRatio', u'capRedu', u'totalCap', u'capCurr', u'parValue', u'annoDate', u'nonRegBegin', u'nonRegEnd', u'subsDateEnd', u'nshareListDate', u'nsharePayDate', u'recordDate', u'lstkNum']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','exDate': 'str','placePriCurr': 'str','rigIssPriCurr': 'str','capCurr': 'str','annoDate': 'str','nonRegBegin': 'str','nonRegEnd': 'str','subsDateEnd': 'str','nshareListDate': 'str','nsharePayDate': 'str','recordDate': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKhaltTEJGet(secID = "", ticker = "", suspDateBegin = "", field = "", pandas = "1"):
    """
    香港上市公司停复牌资料，涵盖暂停交易开始日期，复牌日期，暂停交易原因，暂停交易时段
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param suspDateBegin: 暂停交易开始日期,如'20121214',可空
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
    requestString.append('/api/party/getHKhaltTEJ.csv?ispandas=1&') 
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
    if not isinstance(suspDateBegin, str) and not isinstance(suspDateBegin, unicode):
        suspDateBegin = str(suspDateBegin)

    requestString.append("&suspDateBegin=%s"%(suspDateBegin))
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
            api_base.handle_error(csvString, 'HKhaltTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKhaltTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'suspDateBegin', u'suspRetrad', u'suspReason', u'suspPer']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','suspDateBegin': 'str','suspRetrad': 'str','suspReason': 'str','suspPer': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKmaTEJGet(secID = "", ticker = "", endAccPeriod = "", field = "", pandas = "1"):
    """
    香港上市公司主要销售区域资料，涵盖披露报告月度，营业收入来源国别，该地区收入毛额，收入占公司总营收的比例，营业收入币别
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param endAccPeriod: 披露报告月份,如'201501',可空
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
    requestString.append('/api/party/getHKmaTEJ.csv?ispandas=1&') 
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
    if not isinstance(endAccPeriod, str) and not isinstance(endAccPeriod, unicode):
        endAccPeriod = str(endAccPeriod)

    requestString.append("&endAccPeriod=%s"%(endAccPeriod))
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
            api_base.handle_error(csvString, 'HKmaTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKmaTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'endAccPeriod', u'busiArea', u'income', u'incRatio', u'currencyCd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','endAccPeriod': 'str','busiArea': 'str','currencyCd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKttTEJGet(secID = "", ticker = "", tmpTicker = "", tmpStkName = "", tmpDateBegin = "", field = "", pandas = "1"):
    """
    香港上市公司临时并行交易资料，涵盖临时代码，临时代码简称，临时代码交易单位，临时代码开始日期，临时代码结束日期，临时并行原因，主交易代码，主交易代码旧的交易单位，主交易代码新的交易单位，主交易代码停止交易开始日期，主交易代码停止交易结束日期，主交易代码恢复交易日期，并行买卖开始日期，并行买卖结束日期
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到,可以是列表,secID、ticker至少选择一个
    :param ticker: 港股交易代码,如'00001',可以是列表,secID、ticker至少选择一个
    :param tmpTicker: 临时交易代码,如‘02900’,可空
    :param tmpStkName: 临时代码简称，如‘天安中国’,可空
    :param tmpDateBegin: 临时交易开始时间，如‘20071030’,可空
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
    requestString.append('/api/party/getHKttTEJ.csv?ispandas=1&') 
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
    if not isinstance(tmpTicker, str) and not isinstance(tmpTicker, unicode):
        tmpTicker = str(tmpTicker)

    requestString.append("&tmpTicker=%s"%(tmpTicker))
    if not isinstance(tmpStkName, str) and not isinstance(tmpStkName, unicode):
        tmpStkName = str(tmpStkName)

    requestString.append("&tmpStkName=%s"%(tmpStkName))
    if not isinstance(tmpDateBegin, str) and not isinstance(tmpDateBegin, unicode):
        tmpDateBegin = str(tmpDateBegin)

    requestString.append("&tmpDateBegin=%s"%(tmpDateBegin))
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
            api_base.handle_error(csvString, 'HKttTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKttTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'tmpTicker', u'tmpStkName', u'tmpStkTradingUnit', u'tmpDateBegin', u'tmpDateEnd', u'tmpParReason', u'ticker', u'tradingUnitOld', u'tradingUnitNew', u'oriStradBegin', u'oriStradEnd', u'oriRetradDate', u'parDateBegin', u'parDateEnd']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','tmpTicker': 'str','tmpStkName': 'str','tmpDateBegin': 'str','tmpDateEnd': 'str','tmpParReason': 'str','ticker': 'str','tradingUnitOld': 'str','tradingUnitNew': 'str','oriStradBegin': 'str','oriStradEnd': 'str','oriRetradDate': 'str','parDateBegin': 'str','parDateEnd': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKNewsTEJGet(secID, publishDate = "", field = "", pandas = "1"):
    """
    香港公司新闻信息表，涵盖港股的新闻、公告等
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到，如'00001.XHKG',可以是列表
    :param publishDate: 信息发布日期，如‘20150101’,可空
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
    requestString.append('/api/party/getHKNewsTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKNewsTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKNewsTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'publishDate', u'item', u'num', u'content']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','item': 'str','num': 'str','content': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtBSFinaTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股金融业上市公司资产负债表相关数据
    
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
    requestString.append('/api/fundamental/getHKFdmtBSFinaTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtBSFinaTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtBSFinaTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'CReserCB', u'cashDeposOthBFI', u'TDeposFrOthBFI', u'owesCertiGovHK', u'tradingFA', u'fairValueFA', u'derivAssets', u'currTaxAssets', u'inventories', u'RPReceiv', u'clientLoanOthAcc', u'secInvest', u'loansRecInvest', u'AFSFA', u'HTMInvestNC', u'assocEquity', u'subEquity', u'fixedAssets', u'investProp', u'deferTaxAssets', u'intanAssets', u'othAssets', u'TAssets', u'CBBorr', u'depBalFin', u'HKDollar', u'depos', u'CDLiab', u'issueBond', u'RPPayable', u'tradingFL', u'fairValueFL', u'derivLiab', u'taxesPayable', u'deferTaxLiab', u'bankLoansOth', u'preferStkLiab', u'othLiab', u'SBNLiab', u'TLiab', u'commonStock', u'preferStock', u'proceedsNewIssue', u'capitalReserInc', u'reserve', u'HFSAEquity', u'TEquityAttrP', u'minorityInt', u'TShEquity', u'TLiabEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtBSInduTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股一般产业上市公司资产负债表相关数据
    
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
    requestString.append('/api/fundamental/getHKFdmtBSInduTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtBSInduTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtBSInduTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'fixedAssets', u'tollRoadRight', u'investProp', u'develProp', u'hotelProp', u'biolAssetsNC', u'prepaidLeaseNC', u'goodwill', u'intanAssets', u'subEquity', u'NonContrSubEquity', u'assocEquity', u'comEquity', u'fairValueFANC', u'AFSFA', u'HTMInvestNC', u'derivAssetsNC', u'othNCFA', u'othInvestNC', u'deferAssets', u'othNCA', u'TNCA', u'inventories', u'fairValueFA', u'AFSFAC', u'HTMInvestC', u'derivAssetsC', u'othCFA', u'othInvestC', u'AR', u'othReceiv', u'STLoans', u'prepayment', u'prepaidLeaseC', u'cash', u'HFSA', u'othCA', u'TCA', u'TAssets', u'STBorr', u'AP', u'othPayable', u'divPayable', u'expPayable', u'advanceReceipts', u'incomeTaxPayable', u'NCAWithin1Y', u'derivLiab', u'othFLC', u'liabHFSA', u'othCL', u'TCL', u'NCA', u'LTLiab', u'derivLiabNC', u'othFLNC', u'deferCredit', u'reservePens', u'deferTaxLiab', u'lossReser', u'othNCL', u'othLiab', u'TNCL', u'TLiab', u'NA', u'commonStock', u'preferStock', u'proceedsNewIssue', u'capitalReserInc', u'reserve', u'minorityInt', u'HFSAEquity', u'TShEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtBSInsuTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股保险业上市公司资产负债表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00966.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00966',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fundamental/getHKFdmtBSInsuTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtBSInsuTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtBSInsuTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'fixedAssets', u'investProp', u'intanAssets', u'deferAcqCosts', u'assocEquity', u'comEquity', u'subEquity', u'TFinancInvest', u'fairValueFA', u'AFSFA', u'HTMInvestNC', u'loansRecInvest', u'derivAssets', u'othFA', u'purResale', u'NCDepos', u'cash', u'othCashCEquiv', u'refundCapDepos', u'PRRR', u'reinsurAssets', u'unitLinkProdA', u'sepAccAssets', u'deferTaxAssets', u'othAssets', u'LTLoans', u'TAssets', u'insurContrLiab', u'lifInsContrLiab', u'reserUnePrem', u'reserOutstdClaims', u'reserInsuff', u'othInsContrLiab', u'investContrLiab', u'sepAccLiab', u'claimsPayable', u'policyDivPayable', u'reinsurPayable', u'othPayable', u'notesPayable', u'PH', u'deferTaxLiab', u'incomeTaxPayable', u'bankLoansOth', u'premReceivAdva', u'insurProtFund', u'soldForRepurAcc', u'derivLiab', u'bondPayable', u'preferStkLiab', u'othFL', u'othLiab', u'TLiab', u'commonStock', u'preferStock', u'proceedsNewIssue', u'capitalReserInc', u'reserve', u'HFSAEquity', u'TEquityAttrP', u'minorityInt', u'policyHoldEquity', u'TShEquity', u'TLiabEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtBSInveTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股投资业上市公司资产负债表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'06030.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'06030',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fundamental/getHKFdmtBSInveTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtBSInveTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtBSInveTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'fixedAssets', u'investProp', u'prepaidLeaseNC', u'goodwill', u'intanAssets', u'subEquity', u'assocEquity', u'comEquity', u'fairValueFANC', u'AFSFA', u'HTMInvestNC', u'derivAssetsNC', u'othInvestNC', u'clientLRNC', u'loansReceivNC', u'creditCardRecNC', u'hirePurRecNC', u'deferTaxAssets', u'othNCA', u'TNCA', u'fairValueFA', u'AFSFAC', u'HTMInvestC', u'derivAssetsC', u'othInvestC', u'clientLoanReceivC', u'loansReceivC', u'creditCardReceivC', u'hirePurReceivC', u'RPReceiv', u'othReceiv', u'prepayment', u'prepaidLeaseC', u'cash', u'othCA', u'TCA', u'TAssets', u'STBorr', u'issueBond', u'invManagPayable', u'RPPayable', u'othPayable', u'expPayable', u'advanceReceipts', u'incomeTaxPayable', u'NCAWithin1Y', u'fairValueFL', u'derivLiab', u'othCL', u'TCL', u'NCA', u'LTLiab', u'derivLiabNC', u'othFLNC', u'reservePens', u'deferTaxLiab', u'othNCL', u'TNCL', u'TLiab', u'proceedsNewIssue', u'capitalReserInc', u'commonStock', u'preferStock', u'reserve', u'HFSAEquity', u'TEquityAttrP', u'minorityInt', u'TShEquity']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtISFinaTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股金融业上市公司利润表相关数据
    
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
    requestString.append('/api/fundamental/getHKFdmtISFinaTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtISFinaTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtISFinaTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'intIncome', u'intExp', u'NIntIncome', u'commisIncome', u'commisExp', u'NCommisIncome', u'NTradingIncome', u'fairValueFAIncome', u'NFinInvIncome', u'othOperRev', u'revenue', u'operExp', u'NInsurCL', u'befPrepOperProfit', u'loanImpairOthProv', u'assetsImpair', u'goodwillImpair', u'operateProfitTEJ', u'assocProfit', u'comProfit', u'dispFixAssetsGain', u'dispFinAssetsGain', u'FVChgInvProp', u'fixAssetOthRevDecr', u'assetsImpairAftOP', u'invImpairAftOP', u'othNonOperIncome', u'TProfit', u'incomeTax', u'incomeAftTax', u'gainDiscontOper', u'extraItem', u'cumuEffects', u'depAmor', u'othAI', u'NIncome', u'minorityGain', u'NIncomeAttrP', u'basicEPS', u'sharesW', u'preferDivCurr', u'dilutedEPS', u'sharesDiluW', u'operateProfitREP']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtISInduTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股一般产业上市公司利润表相关数据
    
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
    requestString.append('/api/fundamental/getHKFdmtISInduTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtISInduTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtISInduTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'revenue', u'salesCosts', u'othOperCostsTax', u'grossIncomeTEJ', u'othOperIncome', u'operExpProm', u'operExpAdmin', u'operExpRD', u'othOperExp', u'restrCosts', u'operateProfitTEJ', u'finanIncome', u'finanCost', u'assocProfit', u'comProfit', u'OPAftOthIncome', u'OPAftOthExp', u'TProfit', u'IncomeTax', u'incomeAftTax', u'gainDiscontOper', u'extraItem', u'cumuEffects', u'depAmor', u'othAI', u'NIncome', u'minorityGain', u'NIncomeAttrP', u'basicEPS', u'sharesW', u'preferDivCurr', u'dilutedEPS', u'sharesDiluW', u'EBITDA', u'grossIncomeREP', u'operateProfitREP']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtISInsuTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股保险业上市公司利润表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00966.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00966',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fundamental/getHKFdmtISInsuTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtISInsuTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtISInsuTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'grossPremWrit', u'reinsur', u'NGrossPremWrit', u'unePremReser', u'premEarned', u'InvIncome', u'reinsCommIncome', u'forexGain', u'gainAssoc', u'gainJoint', u'othOperIncome', u'bankIncome', u'revenue', u'insBenefClaimExp', u'policyDiv', u'PHDeposIntExp', u'chgInsurReser', u'invContrExp', u'operProfBefFC', u'commisExp', u'bankExp', u'adminExp', u'incrInsurFund', u'othExp', u'operateProfitTEJ', u'assocProfit', u'comProfit', u'finanCost', u'OPAftOthIncome', u'OPAftOthExp', u'TProfit', u'incomeTax', u'incomeAftTax', u'gainDiscontOper', u'extraItem', u'cumuEffects', u'othAI', u'depAmor', u'NIncome', u'NIncomeAttrP', u'minorityGain', u'policyHoldInt', u'NIncomeDiluted', u'basicEPS', u'sharesW', u'dilutedEPS', u'sharesDiluW', u'preferDivCurr', u'operateProfitREP', u'forexDiffer', u'chgFVAFS', u'CFHedges', u'chgFVOthInv', u'FVGainConvBonds', u'dispAFSFA', u'gainChgInvProp', u'gainChgAssets', u'shareAssoc', u'shareCom', u'shareInvEquity', u'esShareOptionExp', u'actuGainDefiBene', u'othComprIncome', u'incomeTaxComp', u'deferTaxComp', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS', u'CIPolicyHolders']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtISInveTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股投资业上市公司利润表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'06030.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'06030',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fundamental/getHKFdmtISInveTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtISInveTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtISInveTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'revenue', u'intIncome', u'intExp', u'divIncome', u'NAFSFAIncome', u'NFVISIncome', u'FVGainInvProp', u'FVGainFA', u'chgFVISIncome', u'FVGainDerivFA', u'othOperRev', u'operExp', u'befPrepOperProfit', u'loanImpairOthProv', u'operateProfit', u'assocProfit', u'comProfit', u'FVChgInvProp', u'dispFixAssetsGain', u'dispFinAssetsGain', u'OPAftOthIncome', u'OPAftOthExp', u'TProfit', u'incomeTax', u'incomeAftTax', u'gainDiscontOper', u'extraItem', u'depAmor', u'othAI', u'NIncome', u'minorityGain', u'NIncomeAttrP', u'basicEPS', u'sharesW', u'preferDivCurr', u'dilutedEPS', u'sharesDiluW', u'operateProfitREP', u'forexDiffer', u'chgFVAFS', u'CFHedges', u'chgFVOthInv', u'FVGainConvBonds', u'dispAFSFA', u'gainChgInvProp', u'gainChgAssets', u'shareAssoc', u'shareCom', u'shareInvEquity', u'esShareOptionExp', u'actuGainDefiBene', u'othComprIncome', u'incomeTaxComp', u'deferTaxComp', u'TComprIncome', u'comprIncAttrP', u'comprIncAttrMS']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtCFFinaTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股金融业上市公司现金流量表相关数据
    
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
    requestString.append('/api/fundamental/getHKFdmtCFFinaTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtCFFinaTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtCFFinaTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'TProfit', u'NCExtraItem', u'depreciation', u'amortisation', u'intExpOper', u'intIncomeOper', u'CDivFrEquityOper', u'assocProfit', u'assocLoss', u'dispFixAssetsLoss', u'dispFiLoss', u'dispLoanLoss', u'dispIPLoss', u'NEITradingInc', u'loanImpairOthProv', u'accrImpairDisc', u'NAdvaWritRecover', u'fixAssetsImpair', u'FAImpair', u'goodwillImpair', u'othAssetsImpair', u'FVChgInvestProp', u'FVLossOthAssets', u'revalDecrFixAssets', u'FVLossInvDeriv', u'FVGainCertiDepos', u'FVGainFL', u'FVGainOthLiab', u'esShareOptionExp', u'actuGainDefiBene', u'forexGain', u'decrCReserCB', u'CSTFunds', u'chgPlacementBFI', u'chgTradeBillsCD', u'chgTB3MABV', u'chgSecHDP', u'decrFAFVOper', u'decrDerivFAOper', u'chgSecHIP', u'decrAFSFAOper', u'decrHIMFAOper', u'decrLROthFA', u'chgAdvanceCust', u'chgAdvanceBFI', u'chgIntReceivOA', u'invenDecr', u'decrRPReceiv', u'decrDeferAssets', u'decrOthAssets', u'incrLiabToCB', u'chgDeposLoanCust', u'chgCDIssued', u'chgBond', u'chgDeposLoanBank', u'incrFLFV', u'incrDerivFL', u'incrRPPayable', u'incrTradingFL', u'chgIntPayableOth', u'incrDeferTax', u'othLiab', u'forexEffectsOper', u'ANOCF', u'intReceivedOper', u'divReceivedOper', u'payableDivOper', u'interestPaid', u'taxPaid', u'taxPRF', u'contriReserve', u'payPurShareAward', u'NCFOperateA', u'purFixAssets', u'dispFixAssets', u'purInvestProp', u'dispInvestProp', u'purIntanAssets', u'dispIntanAssets', u'purSub', u'dispSub', u'purAssoc', u'dispAssoc', u'incrOthInvest', u'dispOthLTInvest', u'CFFrSLPortf', u'loanSub', u'refundLoanFrSub', u'loanAssoc', u'refundLoanFrAssoc', u'intExpInvest', u'intReceivedInvest', u'divReceivedInvest', u'divAssocInvest', u'ANICF', u'NCFFrInvestA', u'newLTLoan', u'refundLTLoan', u'issueBond', u'refundBond', u'issueCD', u'redeemCD', u'deposAccFrSub', u'deposRefundToSub', u'contriMSH', u'CFFrIncrCapital', u'incrCapitalExp', u'CFIssWarr', u'proceedIssExercWarr', u'incrTreasurStock', u'subordLoanCapiIssue', u'refundSubordLoanCapi', u'CDivFrEquityFinan', u'disDivFinan', u'divPaidMSH', u'preferDivPaid', u'intExpFinan', u'ANFCF', u'NCFFrFinanA', u'NChangeInCashTEJ', u'NCEBegBal', u'forexEffects', u'contrANC', u'NCEEndBal', u'NChangeInCashREP']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtCFInduTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股一般产业上市公司现金流量表相关数据
    
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
    requestString.append('/api/fundamental/getHKFdmtCFInduTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtCFInduTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtCFInduTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'TProfit', u'NCExtraItem', u'finanCost', u'intIncomeOper', u'depreciation', u'amortisation', u'assocProfit', u'assocLoss', u'CDivFrEquityOper', u'esShareOptionExp', u'fixAssetsImpair', u'FAImpair', u'goodwillImpair', u'othAssetsImpair', u'provBadDoubtDebts', u'writOffPIReceiv', u'invenProv', u'FVChgInvestProp', u'revalDecrFixAssets', u'FAFVLoss', u'FVLossDerivFA', u'FVLossConvBonds', u'FVLossOthAssets', u'FVGainFL', u'FVGainDerivFL', u'dispFixAssetsLoss', u'LTInvestLoss', u'dispFAFVLoss', u'dispAFSFALoss', u'dispHTMFALoss', u'dispDerivFALoss', u'forexGain', u'decrAR', u'decrOthReceiv', u'invenDecr', u'decrPrepayment', u'decrFAFVOper', u'decrAFSFAOper', u'decrHIMFAOper', u'decrDerivFAOper', u'decrOthFA', u'incrAP', u'incrOthReceiv', u'incrExpPayable', u'incrAdvaReceipts', u'incrFLFVOper', u'incrDerivFLOper', u'othFL', u'forexEffectsOper', u'ANOCF', u'divReceivedOper', u'intReceivedOper', u'CFFrDispFA', u'payableDivOper', u'interestPaid', u'taxPaid', u'taxPRF', u'NCFOperateA', u'dispFixAssets', u'purFixAssets', u'dispInvestProp', u'purInvestProp', u'dispDeveProp', u'purDeveProp', u'dispIntanAssets', u'purIntanAssets', u'dispSub', u'purSub', u'dispAssoc', u'purAssoc', u'decrFAFVInvest', u'decrAFSFAInvest', u'decrHIMFAInvest', u'dispOthLTInvest', u'incrOthInvest', u'decrDerivFAInvest', u'NCIFNCSub', u'incrFLFVInvest', u'incrDerivFLInvest', u'incrOthFL', u'intExpInvest', u'intReceivedInvest', u'divReceivedInvest', u'divAssocInvest', u'forexAInvest', u'ANICF', u'NCFFrInvestA', u'newSTLoan', u'issueBond', u'refundBond', u'newLTLoan', u'refundLTLoan', u'loanFrMSH', u'refundMSH', u'contriMSH', u'CFIssWarr', u'CFFrIncrCapital', u'incrCapitalExp', u'proceedIssExercWarr', u'incrTreasurStock', u'disDivFinan', u'divPaidMSH', u'preferDivPaid', u'LAInterest', u'intExpFinan', u'ANFCF', u'NCFFrFinanA', u'NChangeInCashTEJ', u'NCEBegBal', u'forexEffects', u'contrANC', u'NCEEndBal', u'NChangeInCashREP']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtCFInsuTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股保险业上市公司现金流量表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'00966.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'00966',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fundamental/getHKFdmtCFInsuTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtCFInsuTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtCFInsuTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'TProfit', u'NCExtraItem', u'depAmor', u'depreciation', u'amortisation', u'FIAmor', u'debtDiscPremAmor', u'assocProfit', u'assocLoss', u'gainDiluInvAssoc', u'FIIncomeOper', u'CDivFrEquityOper', u'intIncome', u'rentReceived', u'unrealisedLossInv', u'realisedGainInv', u'FAImpair', u'othFAIncome', u'intExpOper', u'dispFixAssetsLoss', u'dispIPLoss', u'dispSubLoss', u'dispAssocLoss', u'fixAssetsImpair', u'intanAssetsImpair', u'goodwillImpair', u'assocImpair', u'impairPROper', u'othAssetsImpair', u'revalDecrFixAssets', u'FVLossOthAssets', u'FVChgInvestProp', u'FVGainFL', u'FVGainOthLiab', u'esShareOptionExp', u'actuGainDefiBene', u'forexGain', u'CSTFunds', u'decrFI', u'decrFAFVOper', u'decrAFSFAOper', u'decrHIMFAOper', u'decrLoanOthFA', u'decrDerivFAOper', u'decrOthFA', u'decrPurResOper', u'decrDeferAssets', u'decrOthAssets', u'SFRAccOper', u'incrDeferTax', u'decrPRReinsReceiv', u'decrReinsurContr', u'decrPHAccULP', u'decrDeferAcquiCosts', u'incrInsurContrLiab', u'incrReserLinsLiab', u'incrReserUnePrem', u'incrReserOutstdClaims', u'incrReserInsuffPrem', u'incrOthInsuContr', u'decrInvContrLiab', u'incrClaimPayable', u'incrPHDivPayable', u'incrReinsurPayable', u'incrPHDeposOper', u'incrInsurProt', u'incrFLFV', u'incrDerivFL', u'othFL', u'prAdvaOper', u'othLiab', u'forexEffectsOper', u'divReceivedOper', u'intReceivedOper', u'CFFrDispFA', u'payableDivOper', u'interestPaid', u'contriReserve', u'payPurShareAward', u'taxPaid', u'taxPRF', u'ANOCF', u'NCFOperateA', u'purFixAssets', u'dispFixAssets', u'purInvestProp', u'dispInvestProp', u'purIntanAssets', u'goodwillIncr', u'dispIntanAssets', u'goodwillDecr', u'purSub', u'dispSub', u'purAssoc', u'dispAssoc', u'incrOthInvest', u'purFAFV', u'purAFSFA', u'purHTMFA', u'incrDerivFA', u'purLoanAdva', u'purOthFA', u'dispOthLTInvest', u'dispAFSFA', u'dispFAFV', u'dispHTMFA', u'decrDerivFAInvest', u'CFSaleLoanAdva', u'dispOthFA', u'intExpInvest', u'intReceivedInvest', u'divReceivedInvest', u'divAssocInvest', u'decrPurFAInvest', u'incrSodlFAInvest', u'rentReceivedInvest', u'decrNCDeposInvest', u'ANICF', u'NCFFrInvestA', u'newLTLoan', u'refundLTLoan', u'contriMSH', u'subordLoanCapiIssue', u'refundSubordLoanCapi', u'issueBond', u'refundBond', u'CFFrIncrCapital', u'CFIssWarr', u'proceedIssExercWarr', u'incrCapitalExp', u'incrTreasurStock', u'CDivFrEquityFinan', u'disDivFinan', u'divPaidMSH', u'preferDivPaid', u'intExpFinan', u'decrPurFAFinan', u'incrSodlFAFinan', u'decrNCDeposFinan', u'incrPHDeposFinan', u'ANFCF', u'NCFFrFinanA', u'NChangeInCashTEJ', u'NCEBegBal', u'forexEffects', u'contrANC', u'NCEEndBal', u'NChangeInCashREP']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def HKFdmtCFInveTEJGet(secID = "", ticker = "", publishDateBegin = "", publishDateEnd = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取港股投资业上市公司现金流量表相关数据
    
    :param secID: 证券内部编码,可通过交易代码和交易市场在DataAPI.SecIDGet获取到,如'06030.XHKG',可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,如'06030',可以是列表,secID、ticker至少选择一个
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
    requestString.append('/api/fundamental/getHKFdmtCFInveTEJ.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, 'HKFdmtCFInveTEJGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, 'HKFdmtCFInveTEJGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'repTypeDate', u'mergedFlag', u'fiscalEndDate', u'publishDate', u'startDate', u'endDate', u'fiscalPeriod', u'currencyCD', u'TProfit', u'NCExtraItem', u'finanCost', u'intIncomeOper', u'amortisation', u'depreciation', u'assocProfit', u'assocLoss', u'CDivFrEquityOper', u'esShareOptionExp', u'provBadDoubtDebts', u'fixAssetsImpair', u'goodwillImpair', u'FAImpair', u'othAssetsImpair', u'FVChgInvestProp', u'revalDecrFixAssets', u'FAFVLoss', u'FVLossDerivFA', u'FVLossOthAssets', u'FVGainFL', u'FVGainDerivFL', u'dispFixAssetsLoss', u'dispIPLoss', u'LTInvestLoss', u'dispFAFVLoss', u'dispAFSFALoss', u'dispHTMFALoss', u'dispDerivFALoss', u'forexGain', u'decrClientLR', u'decrLoanReceiv', u'decrCredCardReceiv', u'decrHirePurReceiv', u'decrRPReceiv', u'decrOthReceiv', u'decrPrepayment', u'decrFAFVOper', u'decrAFSFAOper', u'decrHIMFAOper', u'decrDerivFAOper', u'decrOthFA', u'incrInvManPayable', u'incrRPPayable', u'incrOthReceiv', u'incrExpPayable', u'incrAdvaReceipts', u'incrFLFVOper', u'incrDerivFLOper', u'othFL', u'forexEffectsOper', u'ANOCF', u'intReceivedOper', u'divReceivedOper', u'payableDivOper', u'interestPaid', u'taxPaid', u'taxPRF', u'NCFOperateA', u'dispFixAssets', u'purFixAssets', u'dispInvestProp', u'purInvestProp', u'dispIntanAssets', u'purIntanAssets', u'dispSub', u'purSub', u'dispAssoc', u'purAssoc', u'decrFAFVInvest', u'decrAFSFAInvest', u'decrHIMFAInvest', u'dispOthLTInvest', u'incrOthInvest', u'decrDerivFAInvest', u'incrFLFVInvest', u'incrDerivFLInvest', u'incrOthFL', u'intExpInvest', u'intReceivedInvest', u'divReceivedInvest', u'divAssocInvest', u'forexAInvest', u'ANICF', u'NCFFrInvestA', u'newSTLoan', u'newLTLoan', u'refundLTLoan', u'loanFrMSH', u'refundMSH', u'contriMSH', u'issueBond', u'refundBond', u'CFIssWarr', u'CFFrIncrCapital', u'incrCapitalExp', u'proceedIssExercWarr', u'incrTreasurStock', u'disDivFinan', u'divPaidMSH', u'preferDivPaid', u'LAInterest', u'intExpFinan', u'ANFCF', u'NCFFrFinanA', u'NChangeInCashTEJ', u'NCEBegBal', u'forexEffects', u'contrANC', u'NCEEndBal', u'NChangeInCashREP']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','repTypeDate': 'str','mergedFlag': 'str','fiscalEndDate': 'str','fiscalPeriod': 'str','currencyCD': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

