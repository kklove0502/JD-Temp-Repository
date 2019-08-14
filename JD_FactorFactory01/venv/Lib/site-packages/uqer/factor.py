#!coding=utf-8
import datetime

def get_trade_date_list(begin_date, end_date):
    from . import DataAPI
    try:
        df = DataAPI.TradeCalGet(exchangeCD='XSHG', beginDate=begin_date, endDate=end_date, field='calendarDate,isOpen')
        date_list = list(df[df['isOpen'] == 1]['calendarDate'])
        return date_list
    except:
        print ('获取交易日历异常，请检查TradeCalGet的访问权限或者重试')
        return []

def check_sec_id(sec_id):
    if len(sec_id) == 11:
        if not sec_id.endswith('XSHG') and not sec_id.endswith('XSHE'):
            return False
        for i in range(6):
            if not sec_id[i].isdigit() and sec_id[i] != 'T':
                return False
        return True
    elif len(sec_id) == 9:
        if not sec_id.endswith('SZ') and not sec_id.endswith('SH'):
            return False
        if sec_id[0] == 'T':
            pass
        for i in range(6):
            if not sec_id[i].isdigit() and sec_id[i] != 'T':
                return False
        return True
    elif len(sec_id) == 6:
        for i in range(6):
            if not sec_id[i].isdigit() and sec_id[i] != 'T':
                return False
        return True
    return False

def load_df_from_file(file_path):
    try:
        import os
        if not os.path.exists(file_path):
            return None, 'csv文件不存在'
        import pandas as pd
        return pd.read_csv(file_path), ''
    except:
        return None, '错误的csv文件'

def load_df_from_data(data):
    try:
        import StringIO
        import pandas as pd
        return pd.read_csv(StringIO.StringIO(data)), ''
    except:
        return None, '错误的csv文件'

def check_trade_date_format(format, date_string):
    try:
        return datetime.datetime.strptime(str(date_string), format)
    except:
        return None

def convert_one_trade_date(date):
    dt = check_trade_date_format('%Y-%m-%d', date)
    if dt is not None:
        return dt
    dt = check_trade_date_format('%Y%m%d', date)
    if dt is not None:
        return dt
    dt = check_trade_date_format('%Y/%m/%d', date)
    if dt is not None:
        return dt
    return dt

def convert_trade_list(trade_date_list):
    result = []
    for date in trade_date_list:
        dt = convert_one_trade_date(date)
        if dt is not None:
            result.append(dt.strftime('%Y-%m-%d'))
        else:
            print ('交易日不合理：%s' % date)
            return []
    return result

def check_factor_data(df):
    trade_date_column_name = df.columns[0]
    trade_date_list = list(df[trade_date_column_name])
    trade_date_list = convert_trade_list(trade_date_list)
    if not trade_date_list:
        return False, '交易日检查未通过'
    # begin_date = trade_date_list[0]
    # end_date = trade_date_list[-1]
    # standard_date_list = get_trade_date_list(begin_date, end_date)
    # for idx, date in enumerate(trade_date_list):
    #     if date != standard_date_list[idx]:
    #         return False, '交易日不连续：%s' % (date)
    columns = list(df.columns)
    universe_len = len(columns) - 1
    for col in columns:
        if col != trade_date_column_name:
            if not check_sec_id(col):
                return False, '证券代码错误：%s' % col
    # nan_count = df.drop(trade_date_column_name, 1).isnull().sum(axis=1)
    # total = len(trade_date_list) * universe_len
    # none_count = 0
    # for idx, count in nan_count.iteritems():
    #     none_count += count
    #     if count > universe_len-300:
    #         return False, '交易日%s的因子覆盖率过低，覆盖数不到300' % (trade_date_list[idx])
    #
    # none_rate = float(float(none_count) / float(total))
    # if none_rate > 0.7:
    #     return False, '因子覆盖率过低：%0.2f%%，要求不低于30%%' % ((1-none_rate)*100)
    return True, ''