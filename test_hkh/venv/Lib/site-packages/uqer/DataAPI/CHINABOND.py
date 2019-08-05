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

__doc__="中国债券信息网"
def __YieldCurveGet(CurveName, Tenor, CurveType = "", BeginDate = "", EndDate = "", field = "", pandas = "1"):
    """
    输入中债收益率曲线名称、曲线类型、待偿期限，可查询到中债在这些条件下的收益率
    
    :param CurveName: 收益率曲线名称，输入如中债收益率曲线名称“中国固定利率国债收益率曲线”,可同时输入多个收益率曲线名称,可以是列表
    :param Tenor: 待偿期限，以年为单位，如1个月，输入"0.08",,可同时输入多个待偿期限,可以是列表
    :param CurveType: 收益率曲线类型，可选：到期、即期、远期的即期、远期的到期，默认为"到期",可空
    :param BeginDate: 数据起始日期，默认今天，输入格式“YYYYMMDD”,可空
    :param EndDate: 数据起始日期，默认今天，输入格式“YYYYMMDD”,可空
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
    requestString.append('/api/bond/getYieldCurve.csv?ispandas=1&') 
    requestString.append("CurveName=")
    if hasattr(CurveName,'__iter__') and not isinstance(CurveName, str):
        if len(CurveName) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = CurveName
            requestString.append(None)
        else:
            requestString.append(','.join(CurveName))
    else:
        requestString.append(CurveName)
    requestString.append("&Tenor=")
    if hasattr(Tenor,'__iter__') and not isinstance(Tenor, str):
        if len(Tenor) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = Tenor
            requestString.append(None)
        else:
            requestString.append(','.join(Tenor))
    else:
        requestString.append(Tenor)
    if not isinstance(CurveType, str) and not isinstance(CurveType, unicode):
        CurveType = str(CurveType)

    requestString.append("&CurveType=%s"%(CurveType))
    try:
        BeginDate = BeginDate.strftime('%Y%m%d')
    except:
        BeginDate = BeginDate.replace('-', '')
    requestString.append("&BeginDate=%s"%(BeginDate))
    try:
        EndDate = EndDate.strftime('%Y%m%d')
    except:
        EndDate = EndDate.replace('-', '')
    requestString.append("&EndDate=%s"%(EndDate))
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
            api_base.handle_error(csvString, '__YieldCurveGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__YieldCurveGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'CurveName', u'CurveType', u'Tenor', u'DataDate', u'Yield']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'CurveName': 'str','CurveType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __CBBenchmarkIndicesGet(benchmarkName = "", valuationType = "", beginDate = "", endDate = "", field = "", pandas = "1"):
    """
    获取中债指数各代偿期财富指数、净价指数、全价指数、平均市值法久期、平均市值法凸性等信息
    
    :param benchmarkName: 中债指数指数名称,可以是列表,benchmarkName、valuationType、beginDate、endDate至少选择一个
    :param valuationType: 如以下：10年以上；1-3年；1年以下；3-5年；5-7年；7-10年；总值,可以是列表,benchmarkName、valuationType、beginDate、endDate至少选择一个
    :param beginDate: 查询起始时间，格式为YYYYMMDD,benchmarkName、valuationType、beginDate、endDate至少选择一个
    :param endDate: 查询截止时间，格式为YYYYMMDD,benchmarkName、valuationType、beginDate、endDate至少选择一个
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
    requestString.append('/api/idxMarket/getCBBenchmarkIndices.csv?ispandas=1&') 
    requestString.append("benchmarkName=")
    if hasattr(benchmarkName,'__iter__') and not isinstance(benchmarkName, str):
        if len(benchmarkName) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = benchmarkName
            requestString.append(None)
        else:
            requestString.append(','.join(benchmarkName))
    else:
        requestString.append(benchmarkName)
    requestString.append("&valuationType=")
    if hasattr(valuationType,'__iter__') and not isinstance(valuationType, str):
        if len(valuationType) > 100 and split_param is None:
            split_index = len(requestString)
            split_param = valuationType
            requestString.append(None)
        else:
            requestString.append(','.join(valuationType))
    else:
        requestString.append(valuationType)
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
            api_base.handle_error(csvString, '__CBBenchmarkIndicesGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__CBBenchmarkIndicesGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'dataDate', u'benchmarkName', u'valuationTypeCD', u'valuationType', u'weightedAvgMktCap', u'weightedAvgGrossPx', u'weightedAvgNetPx', u'mktValDura', u'cashflowDiscDura', u'mktValConv', u'cashflowDiscConv', u'yieldToMaturity', u'baseRate', u'yearsToMaturity', u'weightedAvgCoupon', u'indexValue', u'mktCapIncrement', u'grossPxIncrement', u'netPxIncrement', u'mktValYtm', u'cashbondNav']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'benchmarkName': 'str','valuationTypeCD': 'str','valuationType': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

def __CBBenchmarkCompositionGet(benchmarkName = "", valuationTypeDesc = "", dataDate = "", field = "", pandas = "1"):
    """
    获取中债指数样本券代码、样本券交易市场、样本券净价、全价、到期收益率、指数市值权重等信息
    
    :param benchmarkName: 中债指数指数名称,benchmarkName、valuationTypeDesc、dataDate至少选择一个
    :param valuationTypeDesc: 如以下：10年以上；1-3年；1年以下；3-5年；5-7年；7-10年；总值,benchmarkName、valuationTypeDesc、dataDate至少选择一个
    :param dataDate: 日期，格式为YYYYMMDD,benchmarkName、valuationTypeDesc、dataDate至少选择一个
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
    requestString.append('/api/idxWeight/getCBBenchmarkComposition.csv?ispandas=1&') 
    if not isinstance(benchmarkName, str) and not isinstance(benchmarkName, unicode):
        benchmarkName = str(benchmarkName)

    requestString.append("benchmarkName=%s"%(benchmarkName))
    if not isinstance(valuationTypeDesc, str) and not isinstance(valuationTypeDesc, unicode):
        valuationTypeDesc = str(valuationTypeDesc)

    requestString.append("&valuationTypeDesc=%s"%(valuationTypeDesc))
    try:
        dataDate = dataDate.strftime('%Y%m%d')
    except:
        dataDate = dataDate.replace('-', '')
    requestString.append("&dataDate=%s"%(dataDate))
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
            api_base.handle_error(csvString, '__CBBenchmarkCompositionGet')
        elif csvString[:2] == '-1':
            csvString = ''
    else:
        p_list = api_base.splist(split_param, 100)
        csvString = []
        for index, item in enumerate(p_list):
            requestString[split_index] = ','.join(item)
            temp_result = api_base.__getCSV__(''.join(requestString), httpClient, gw=True)
            if temp_result is None or len(temp_result) == 0 or temp_result[0] == '{' or (temp_result[0] == '-' and not api_base.is_no_data_warn(temp_result, False)):
                api_base.handle_error(temp_result, '__CBBenchmarkCompositionGet')
            if temp_result[:2] != '-1':
                csvString.append(temp_result if len(csvString) == 0 else temp_result[temp_result.find('\n')+1:])
        csvString = ''.join(csvString)
    
    if len(csvString) == 0:
        if 'field' not in locals() or len(field) == 0:
            field = [u'dataDate', u'benchmarkName', u'valuationTypeCD', u'valuationTypeDesc', u'xibeTicker', u'bondShortName', u'marketCD', u'market', u'grossPx', u'netPx', u'accruedInterest', u'increment', u'yearsToMaturity', u'yieldToMaturity', u'spread', u'modifiedDuration', u'spotDuration', u'spreadDuratoin', u'convexity', u'spotConvexity', u'spreadConvexity', u'baseRate', u'indexWeight']
        if hasattr(field, '__iter__') and not isinstance(field, str):
            csvString = ','.join(field) + '\n'
        else:
            csvString = field + '\n'
    if pandas != "1":
        put_data_in_cache(func_name, cache_key, csvString)
        return csvString
    try:
        myIO = StringIO(csvString)
        pdFrame = pd.read_csv(myIO, dtype = {'benchmarkName': 'str','valuationTypeCD': 'str','valuationTypeDesc': 'str','xibeTicker': 'str','bondShortName': 'str','marketCD': 'str','market': 'str'},  )
        put_data_in_cache(func_name, cache_key, pdFrame)
        return pdFrame
    except Exception as e:
        raise e
    finally:
        myIO.close()

