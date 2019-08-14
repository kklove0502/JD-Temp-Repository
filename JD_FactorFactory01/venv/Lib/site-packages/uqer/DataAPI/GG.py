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

__doc__="朝阳永续"
def __ReportGGGet(secID = "", ticker = "", writeDate = "", orgName = "", BeginWriteDate = "", EndwriteDate = "", title = "", author = "", field = "", pandas = "1"):
    """
    朝阳永续数据库的研报基本信息以及预测信息，历史数据从2011年开始，覆盖A股及港股信息，内容包含公司EPS、PE、营业收入、净资产收益率、EV_EBITDA、净利润、评级以及目标价等预测信息。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker、writeDate至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker、writeDate至少选择一个
    :param writeDate: 书写日期,输入格式“YYYYMMDD”,secID、ticker、writeDate至少选择一个
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
    requestString.append('/api/report/getReportGG.csv?ispandas=1&') 
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
        writeDate = writeDate.strftime('%Y%m%d')
    except:
        writeDate = writeDate.replace('-', '')
    requestString.append("&writeDate=%s"%(writeDate))
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
            api_base.handle_error(csvString, '__ReportGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__ReportGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'orgName', u'writeDate', u'secID', u'ticker', u'SType', u'secShortName', u'title', u'author', u'ForecastPeriodTime', u'ADJEPS', u'PE', u'Sales', u'PnetProfit', u'ROE', u'OprProfit', u'EV_EBITDA', u'Recommadation', u'ADJRecommadation', u'targetPrice', u'Content']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'orgName': 'str','secID': 'str','ticker': 'str','SType': 'str','secShortName': 'str','title': 'str','author': 'str','Recommadation': 'str','ADJRecommadation': 'str','Content': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __ReportadjustGGGet(ticker = "", RRID = "", currentCreateDate = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    朝阳永续对研报预测净利润、EPS等信息的调整数据
    
    :param ticker: 证券代码,可以是列表,ticker、RRID、currentCreateDate至少选择一个
    :param RRID: 研报ID,可以是列表,ticker、RRID、currentCreateDate至少选择一个
    :param currentCreateDate: 本次预测日期,输入格式“YYYYMMDD”,ticker、RRID、currentCreateDate至少选择一个
    :param beginDate: 查询开始日期，输入格式“YYYYMMDD”,可空
    :param endDate: 查询结束日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/report/getReportadjustGG.csv?ispandas=1&') 
    requestString.append("ticker=")
    if hasattr(ticker,'__iter__') and not isinstance(ticker, str):
        if len(ticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ticker
            requestString.append(None)
        else:
            requestString.append(','.join(ticker))
    else:
        requestString.append(ticker)
    requestString.append("&RRID=")
    if hasattr(RRID,'__iter__') and not isinstance(RRID, str):
        if len(RRID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = RRID
            requestString.append(None)
        else:
            requestString.append(','.join(RRID))
    else:
        requestString.append(RRID)
    try:
        currentCreateDate = currentCreateDate.strftime('%Y%m%d')
    except:
        currentCreateDate = currentCreateDate.replace('-', '')
    requestString.append("&currentCreateDate=%s"%(currentCreateDate))
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
            api_base.handle_error(csvString, '__ReportadjustGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__ReportadjustGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'RRID', u'ticker', u'orgName', u'forecastYear', u'currentCreateDate', u'previousCreateDate', u'currentForecastProfit', u'previousForecastProfit', u'currentForecastEPS', u'previousForecastEPS', u'profitAdjustFlag', u'intoDate', u'profitAdjustRate', u'EPSAdjustRate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'RRID': 'str','ticker': 'str','orgName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __DerReportGGGet(secID = "", ticker = "", createDate = "", RRID = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    朝阳永续研报基础信息表，包含研报标题、摘要、研究员、研究机构以及评级等信息。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker、createDate至少选择一个
    :param ticker: 证券代码,可以是列表,secID、ticker、createDate至少选择一个
    :param createDate: 报告撰写日期,输入格式“YYYYMMDD”,secID、ticker、createDate至少选择一个
    :param RRID: 研报ID,可以是列表,可空
    :param beginDate: 查询报告开始日期，输入格式“YYYYMMDD”,可空
    :param endDate: 查询报告结束日期，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/report/getDerReportGG.csv?ispandas=1&') 
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
        createDate = createDate.strftime('%Y%m%d')
    except:
        createDate = createDate.replace('-', '')
    requestString.append("&createDate=%s"%(createDate))
    requestString.append("&RRID=")
    if hasattr(RRID,'__iter__') and not isinstance(RRID, str):
        if len(RRID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = RRID
            requestString.append(None)
        else:
            requestString.append(','.join(RRID))
    else:
        requestString.append(RRID)
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
            api_base.handle_error(csvString, '__DerReportGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__DerReportGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'RRID', u'secID', u'ticker', u'secShortName', u'title', u'content', u'typeID', u'type', u'orgID', u'orgName', u'author', u'scoreCD', u'score', u'organScoreCD', u'orgScore', u'createDate', u'intoDate', u'text1', u'text2', u'targetprice', u'text8', u'priceCurrent', u'capitalCurrent', u'attention', u'attentionName']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','title': 'str','content': 'str','type': 'str','orgName': 'str','author': 'str','score': 'str','orgScore': 'str','text1': 'str','text2': 'str','attentionName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __cmbReportGGGet(secID = "", ticker = "", createDate = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    可获得朝阳永续研报的基本信息和预测信息,覆盖A股和B股。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker、createDate至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker、createDate至少选择一个
    :param createDate: 报告撰写日期,secID、ticker、createDate至少选择一个
    :param beginDate: 报告撰写日期，获得该日期之后撰写的数据，输入格式“YYYYMMDD”,可空
    :param endDate: 报告撰写日期，获得该日期之前撰写的数据，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/report/getcmbReportGG.csv?ispandas=1&') 
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
        createDate = createDate.strftime('%Y%m%d')
    except:
        createDate = createDate.replace('-', '')
    requestString.append("&createDate=%s"%(createDate))
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
            api_base.handle_error(csvString, '__cmbReportGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__cmbReportGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'rrID', u'secID', u'originID', u'ticker', u'secShortName', u'Title', u'content', u'typeID', u'partyID', u'author', u'scoreCD', u'organScoreCD', u'createDate', u'intoDate', u'Text3', u'Text5', u'Text8', u'priceCurrent', u'capitalCurrent', u'Attention', u'attentionName', u'scoreFlag', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','originID': 'str','ticker': 'str','secShortName': 'str','Title': 'str','content': 'str','author': 'str','Text3': 'str','attentionName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __cmbReportSubGGGet(rrID = "", field = "", pandas = "1"):
    """
    获得对应朝阳永续预测表、报告预测数据合并表的预测数据，包含预期营业收入、预期每股收益、预期净利润等预测数据。
    
    :param rrID: 对应朝阳永续预测表、报告预测数据合并表中的rrID,可以是列表,可空
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
    requestString.append('/api/report/getcmbReportSubGG.csv?ispandas=1&') 
    requestString.append("rrID=")
    if hasattr(rrID,'__iter__') and not isinstance(rrID, str):
        if len(rrID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = rrID
            requestString.append(None)
        else:
            requestString.append(','.join(rrID))
    else:
        requestString.append(rrID)
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
            api_base.handle_error(csvString, '__cmbReportSubGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__cmbReportSubGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'rrID', u'originID', u'year', u'quarter', u'foreIncome', u'foreProfit', u'foreIncomeShare', u'foreReturnCashShare', u'foreReturnCapitalShare', u'foreReturn', u'rTar3', u'rTar5', u'profitFlag', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __derReportResGGGet(secID = "", ticker = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    朝阳永续研报基础信息表，包含研报标题、摘要、研究员、研究机构以及评级等信息。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param beginDate: 报告撰写日期，获得该日期之后撰写的数据，输入格式“YYYYMMDD”,可空
    :param endDate: 报告撰写日期，获得该日期之前撰写的数据，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/report/getderReportResGG.csv?ispandas=1&') 
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
            api_base.handle_error(csvString, '__derReportResGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__derReportResGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'rrID', u'secID', u'ticker', u'secShortName', u'title', u'content', u'typeId', u'partyID', u'author', u'scoreCD', u'organScoreCD', u'createTime', u'intoTime', u'text5', u'text8', u'priceCurrent', u'capitalCurrent', u'attention', u'attentionName', u'scoreFlag', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str','title': 'str','content': 'str','author': 'str','attentionName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __derReportSubGGGet(rrID = "", field = "", pandas = "1"):
    """
    获得对应朝阳永续研究报告（个股）的预测数据，包含预期营业收入、预期每股收益、预期净利润等预测数据。
    
    :param rrID: 对应朝阳永续研究报告信息（个股）表中的rrID,可以是列表,可空
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
    requestString.append('/api/report/getderReportSubGG.csv?ispandas=1&') 
    requestString.append("rrID=")
    if hasattr(rrID,'__iter__') and not isinstance(rrID, str):
        if len(rrID) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = rrID
            requestString.append(None)
        else:
            requestString.append(','.join(rrID))
    else:
        requestString.append(rrID)
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
            api_base.handle_error(csvString, '__derReportSubGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__derReportSubGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'rrID', u'year', u'quarter', u'foreIncome', u'foreProfit', u'foreIncomeShare', u'foreReturnCashShare', u'foreReturnCapitalShare', u'foreReturn', u'rTar3', u'rTar5', u'profitFlag', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __CGRDReportGGGet(secID = "", ticker = "", BeginPubDate = "", EndPubDate = "", field = "", pandas = "1"):
    """
    朝阳永续研报一致性预期数据表的评级数据，从2011年开始，内容涵盖A股市场。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param BeginPubDate: 数据发布日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param EndPubDate: 数据发布日期，获得该日期之前发布的数据，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/consensus/getCGRDReportGG.csv?ispandas=1&') 
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
        BeginPubDate = BeginPubDate.strftime('%Y%m%d')
    except:
        BeginPubDate = BeginPubDate.replace('-', '')
    requestString.append("&BeginPubDate=%s"%(BeginPubDate))
    try:
        EndPubDate = EndPubDate.strftime('%Y%m%d')
    except:
        EndPubDate = EndPubDate.replace('-', '')
    requestString.append("&EndPubDate=%s"%(EndPubDate))
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
            api_base.handle_error(csvString, '__CGRDReportGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__CGRDReportGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'SType', u'ticker', u'publishDate', u'secShortName', u'targetPrice', u'targetPriceType', u'Rating', u'RatingType']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','SType': 'str','ticker': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __CESTReportGGGet(secID = "", ticker = "", BeginPubDate = "", EndPubDate = "", field = "", pandas = "1"):
    """
    朝阳永续研报一致性预期数据表中的预测数据，从2011年开始，涵盖A股市场，内容包括EPS、PE、营业收入、净利润等。
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param BeginPubDate: 数据发布日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param EndPubDate: 数据发布日期，获得该日期之前发布的数据，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/consensus/getCESTReportGG.csv?ispandas=1&') 
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
        BeginPubDate = BeginPubDate.strftime('%Y%m%d')
    except:
        BeginPubDate = BeginPubDate.replace('-', '')
    requestString.append("&BeginPubDate=%s"%(BeginPubDate))
    try:
        EndPubDate = EndPubDate.strftime('%Y%m%d')
    except:
        EndPubDate = EndPubDate.replace('-', '')
    requestString.append("&EndPubDate=%s"%(EndPubDate))
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
            api_base.handle_error(csvString, '__CESTReportGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__CESTReportGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'SType', u'ticker', u'publishDate', u'secShortName', u'ForecastPeriodTime', u'EPS_con', u'EPSType', u'Pnetprofit_con', u'PnetprofitType', u'PE_con', u'PEG_con', u'ROE_con', u'STOCK_TYPE', u'CON_HISDATE', u'C4_HISDATE', u'C3', u'C7', u'CB', u'PB_CON', u'INCOME', u'INCOME_TYPE']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','SType': 'str','ticker': 'str','secShortName': 'str','INCOME_TYPE': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __conIncomeGGGet(ticker, beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    朝阳永续对各证券的一致性预期收入的数据信息
    
    :param ticker: 证券代码,可以是列表
    :param beginDate: 查询开始日期,可空
    :param endDate: 查询结束日期,可空
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
    requestString.append('/api/consensus/getconIncomeGG.csv?ispandas=1&') 
    requestString.append("ticker=")
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
            api_base.handle_error(csvString, '__conIncomeGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__conIncomeGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'ticker', u'tdate', u'rptDate', u'incomeType', u'income', u'incomeHisdate']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'ticker': 'str','incomeType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __CESTStockGGGet(secID = "", ticker = "", BeginPubDate = "", EndPubDate = "", field = "", pandas = "1"):
    """
    可获得朝阳永续个股一致预期数据，包含一致预期EPS 、一致预期PE等
    
    :param secID: 研究对象在通联内部的代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param ticker: 研究对象上市代码，可输入多个,可以是列表,secID、ticker至少选择一个
    :param BeginPubDate: 数据发布日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param EndPubDate: 数据发布日期，获得该日期之前发布的数据，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/consensus/getCESTStockGG.csv?ispandas=1&') 
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
        BeginPubDate = BeginPubDate.strftime('%Y%m%d')
    except:
        BeginPubDate = BeginPubDate.replace('-', '')
    requestString.append("&BeginPubDate=%s"%(BeginPubDate))
    try:
        EndPubDate = EndPubDate.strftime('%Y%m%d')
    except:
        EndPubDate = EndPubDate.replace('-', '')
    requestString.append("&EndPubDate=%s"%(EndPubDate))
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
            api_base.handle_error(csvString, '__CESTStockGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__CESTStockGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'SType', u'ticker', u'secShortName', u'stockType', u'publishDate', u'rptDate', u'rptType', u'EPSCon', u'EPSType', u'conHisdate', u'pnetprofitCon', u'pnetprofitType', u'C4Hisdate', u'PECon', u'PEGCon', u'ROECon', u'C3', u'C7', u'C80', u'C81', u'C82', u'C83', u'C84', u'CB', u'PBCon', u'updateTime']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','SType': 'str','ticker': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __SWconForeGGGet(ticker = "", industry = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期（申万行业）数据，包括一致预期EPS，一致预期PE,一致预期ROE，一致预期PB等数据。
    
    :param ticker: 行业代码,可以是列表,ticker、industry至少选择一个
    :param industry: 行业名称,可以是列表,ticker、industry至少选择一个
    :param beginDate: 数据预测日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param endDate: 数据预测日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getSWconForeGG.csv?ispandas=1&') 
    requestString.append("ticker=")
    if hasattr(ticker,'__iter__') and not isinstance(ticker, str):
        if len(ticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ticker
            requestString.append(None)
        else:
            requestString.append(','.join(ticker))
    else:
        requestString.append(ticker)
    requestString.append("&industry=")
    if hasattr(industry,'__iter__') and not isinstance(industry, str):
        if len(industry) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = industry
            requestString.append(None)
        else:
            requestString.append(','.join(industry))
    else:
        requestString.append(industry)
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__SWconForeGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__SWconForeGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'ticker', u'industry', u'tickerType', u'conDate', u'rptDate', u'rptType', u'C1', u'conType', u'C4', u'c4Type', u'C3', u'C5', u'C6', u'C7', u'C12', u'CB', u'CPB', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'ticker': 'str','industry': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __SWconForeC2GGGet(ticker = "", industry = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期衍生指标计算(申万行业)数据，包括滚动EPS，滚动净利润，滚动PE等数据。
    
    :param ticker: 行业代码,可以是列表,ticker、industry至少选择一个
    :param industry: 行业名称,可以是列表,ticker、industry至少选择一个
    :param beginDate: 数据预测日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param endDate: 数据预测日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getSWconForeC2GG.csv?ispandas=1&') 
    requestString.append("ticker=")
    if hasattr(ticker,'__iter__') and not isinstance(ticker, str):
        if len(ticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ticker
            requestString.append(None)
        else:
            requestString.append(','.join(ticker))
    else:
        requestString.append(ticker)
    requestString.append("&industry=")
    if hasattr(industry,'__iter__') and not isinstance(industry, str):
        if len(industry) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = industry
            requestString.append(None)
        else:
            requestString.append(','.join(industry))
    else:
        requestString.append(industry)
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__SWconForeC2GGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__SWconForeC2GGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'ticker', u'industry', u'tickerType', u'conDate', u'C13', u'C9', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'ticker': 'str','industry': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __SWconForeC3GGGet(ticker = "", industry = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期滚动衍生数据表(申万行业)数据，包括滚动净资产，滚动市净率（滚动PB），滚动净利润复合增长率等数据。
    
    :param ticker: 行业代码,可以是列表,ticker、industry至少选择一个
    :param industry: 行业名称,可以是列表,ticker、industry至少选择一个
    :param beginDate: 数据预测日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param endDate: 数据预测日期，获得该日期之后发布的数据，输入格式“YYYYMMDD”,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getSWconForeC3GG.csv?ispandas=1&') 
    requestString.append("ticker=")
    if hasattr(ticker,'__iter__') and not isinstance(ticker, str):
        if len(ticker) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = ticker
            requestString.append(None)
        else:
            requestString.append(','.join(ticker))
    else:
        requestString.append(ticker)
    requestString.append("&industry=")
    if hasattr(industry,'__iter__') and not isinstance(industry, str):
        if len(industry) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = industry
            requestString.append(None)
        else:
            requestString.append(','.join(industry))
    else:
        requestString.append(industry)
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__SWconForeC3GGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__SWconForeC3GGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'ticker', u'industry', u'tickerType', u'conDate', u'CGB', u'CGPB', u'CGG', u'CGPEG', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'ticker': 'str','industry': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __conForeIdxC2GGGet(secID = "", ticker = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳一致预期衍生指标计算(指数)，包括滚动净利润，滚动PE等数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param beginDate: 预测日期（开始日期）,可空
    :param endDate: 预测日期（截止日期）,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getconForeIdxC2GG.csv?ispandas=1&') 
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__conForeIdxC2GGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__conForeIdxC2GGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'tickerType', u'conDate', u'C13', u'C9', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __conForeStkC2GGGet(secID = "", ticker = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期衍生指标计算(个股)，包括滚动EPS，滚动净利润,滚动PE等数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param beginDate: 预测日期（开始日期）,可空
    :param endDate: 预测日期（截止日期）,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getconForeStkC2GG.csv?ispandas=1&') 
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__conForeStkC2GGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__conForeStkC2GGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'conDate', u'C2', u'C13', u'C9', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __conForeIdxC3GGGet(secID = "", ticker = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期滚动衍生数据(指数)，包括滚动净资产，滚动市净率（滚动PB）,滚动净利润复合增长率，滚动PEG等数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param beginDate: 预测日期（开始日期）,可空
    :param endDate: 预测日期（截止日期）,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getconForeIdxC3GG.csv?ispandas=1&') 
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__conForeIdxC3GGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__conForeIdxC3GGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'stockName', u'stockType', u'conDate', u'CGB', u'CGPB', u'CGG', u'CGPEG', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','stockName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __conForeStkC3GGGet(secID = "", ticker = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期滚动衍生数据(个股)，包括滚动净资产，滚动市净率（滚动PB）,滚动净利润复合增长率，滚动PEG等数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param beginDate: 预测日期（开始日期）,可空
    :param endDate: 预测日期（截止日期）,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getconForeStkC3GG.csv?ispandas=1&') 
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__conForeStkC3GGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__conForeStkC3GGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'conDate', u'CGB', u'CGPB', u'CGG', u'CGPEG', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __conForeIdxGGGet(secID = "", ticker = "", beginDate = "", endDate = "", tmStamp = "", field = "", pandas = "1"):
    """
    获得朝阳永续一致预期（指数）数据，包括一致预期EPS，一致预期PE,一致预期ROE，一致预期PB等数据。
    
    :param secID: 证券内部编码，可通过交易代码和交易市场在DataAPI.SecIDGet获取到。,可以是列表,secID、ticker至少选择一个
    :param ticker: 交易代码,可以是列表,secID、ticker至少选择一个
    :param beginDate: 预测日期（开始日期）,可空
    :param endDate: 预测日期（截止日期）,可空
    :param tmStamp: 通过判断下载标识的变动便于及时发现数据的增量及更新。,可空
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
    requestString.append('/api/consensus/getconForeIdxGG.csv?ispandas=1&') 
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
    if not isinstance(tmStamp, str) and not isinstance(tmStamp, unicode):
        tmStamp = str(tmStamp)

    requestString.append("&tmStamp=%s"%(tmStamp))
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
            api_base.handle_error(csvString, '__conForeIdxGGGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__conForeIdxGGGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'secID', u'ticker', u'secShortName', u'conDate', u'c4Type', u'rptDate', u'rptType', u'C1', u'C3', u'C4', u'C5', u'C6', u'C7', u'C12', u'CB', u'CPB', u'tmStamp']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'secID': 'str','ticker': 'str','secShortName': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

