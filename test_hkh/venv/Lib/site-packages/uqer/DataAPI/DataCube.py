# -*- coding: utf-8 -*-
# 通联数据机密
# --------------------------------------------------------------------
# 通联数据股份公司版权所有 © 2013-2017
#
# 注意：本文所载所有信息均属于通联数据股份公司资产。本文所包含的知识和技术概念均属于
# 通联数据产权，并可能由中国、美国和其他国家专利或申请中的专利所覆盖，并受商业秘密或
# 版权法保护。
# 除非事先获得通联数据股份公司书面许可，严禁传播文中信息或复制本材料。
#
# DataYes CONFIDENTIAL
# --------------------------------------------------------------------
# Copyright © 2013-2016 DataYes, All Rights Reserved.
#
# NOTICE: All information contained herein is the property of DataYes
# Incorporated. The intellectual and technical concepts contained herein are
# proprietary to DataYes Incorporated, and may be covered by China, U.S. and
# Other Countries Patents, patents in process, and are protected by trade
# secret or copyright law.
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from DataYes.
'''
Created on 2017年4月28日

@author: jiangtao.sheng
'''

from . data_service.market_service import MarketService
from . data_service.asset_service import AssetService
from . data_service.tick_roller import TickRoller
from . data_service.utils.datetime_utils import get_trading_days
from . data_service.loader.factor_loader import translate_factors
from . data_service.loader.cache_api import get_updated_date, HelloUser, get_td_from_cache
from . api_base import is_pro_user, is_enterprise_user
import datetime
import pandas as pd
import uuid
import time
import os
import numpy as np
from dateutil.parser import parse
from . data_service.const import (EQUITY_DAILY_FIELDS,
                                  STOCK_DAILY_FQ_FIELDS,
                                  EQUITY_MINUTE_FIELDS,
                                  FUTURES_DAILY_FIELDS,
                                  FUTURES_MINUTE_FIELDS,
                                  FS_FIELD,
                                  ALIAS_DAILY_PRICE, ALIAS_MINUTE_PRICE)
MAX_END_DATE = '3000'
MAX_MINUTE_LENGTH = 10000000


MKT_DAILY_FIELDS = set(EQUITY_DAILY_FIELDS) | set(STOCK_DAILY_FQ_FIELDS) \
    | set(FUTURES_DAILY_FIELDS) | set(ALIAS_DAILY_PRICE)

MKT_INTRADAY_FIELDS = set(EQUITY_MINUTE_FIELDS) | set(FUTURES_MINUTE_FIELDS) | \
    set(ALIAS_MINUTE_PRICE)

price_key_mapping = {}


def uniform_date(input_date):
    if isinstance(input_date, basestring):
        format_date = None
        try:
            format_date = parse(input_date)
        except:
            print (u'输入的日期：%s 无法解析，请重新输入' % input_date)
        return format_date
    if isinstance(input_date, datetime.datetime) or isinstance(input_date, datetime.date):
        return input_date
    else:
        print (u'输入的日期格式不支持，请使用 datetime或string类型.')
        return


TODAY = datetime.datetime.today()
TODAY_STR = TODAY.strftime('%Y%m%d')
YESTERDAY = TODAY - datetime.timedelta(days=1)


def get_data_cube(symbol, field, start, end=None, freq='1d', style='sat', adj=None, **kwargs):
    '''
        :param symbol: [list or str] 支持股票、基金、期货（包括连续合约，主力合约、次主力合约）、期权、指数，股票基金。
        :param field: [list or str] 支持行情数据、424因子数据、众筹因子
        :param start： [str or datetime] 返回数据的开始日期，支持"2017-02-02"的字符串格式，也支持Python 中时间和日期格式。
        :param end： [str or datetime] 默认值为当天，返回数据的截止日期，支持"2017-02-02"的字符串格式，也支持Python 中时间和日期格式
        :param freq: [str] 默认值为d。支持日(1d)，多频率的分钟线(1m,5m,15m,30m,60m)
        :param adj: [str] 复权方式 默认为空值，'pre'指代前复权 
        :param style： [str] 数据返回的类型，默认值sat。支持ast/sat/tas，分钟线不支持'tas'。
                                                                其中'a'表示'attribute'、's'表示symbol、't'表示时间。
                        如'ast'就表示返回的Panel中的键是attribute，其值为列为symbol、行为time的DataFrame

    '''
    entry_time = time.time()
    uid = str(uuid.uuid1())
    # 权限控制
    if not (is_pro_user() or is_enterprise_user()):
        print (u"该API仅限优矿专业版、企业版使用，请联系优矿客服 service.uqer@datayes.com，谢谢！")
        return

    # format symbol
    symbol = __format_str_to_arr(symbol, "symbol")
    if not symbol:
        return

    # format field
    field = __format_str_to_arr(field, "field")
    if not field:
        return

    # validate freq
    if isinstance(freq, basestring):
        if freq not in ['1d', '1m', 'd', 'm', '5m', '15m', '30m', '60m']:
            print (
                u"目前只支持'1d','1m','5m', '15m', '30m', '60m'六个频率类型，暂不支持:%s。" % freq)
            return
    else:
        print (u"参数freq的值支持str类型，请校验后重新输入。")

    # validate adj
    if adj is not None and adj != 'pre':
        print (u"目前adj参数只支持'pre'，表示行情前复权，无法识别传入的adj参数值，默认返回未复权行情数据。")

    # parse date
    start_date = uniform_date(start)
    if start_date is None:
        return

    stocks = AssetService.findout_stocks(symbol)
    funds = AssetService.findout_funds(symbol)
    index = AssetService.findout_indexes(symbol)
    futures = AssetService.findout_futures(symbol)

    # 分钟线限制
    if freq in ['1m', '5m', '15m', '30m', '60m', 'm']:
        if (stocks or funds or index) and start_date < datetime.datetime(2009, 1, 1):
            print (
                u"使用分钟线：目前股票、基金、指数支持的最早start为2009-01-01，期货支持的最早start为2010-01-01，自动修改为2009-01-01。")
            start_date = datetime.datetime(2009, 1, 1)
        elif start_date < datetime.datetime(2010, 1, 1):
            print (u"使用分钟线：期货支持的最早start为2010-01-01，自动修改为2010-01-01。")
            start_date = datetime.datetime(2010, 1, 1)

    # validate field
    mkt_daily_fields = set(field) & MKT_DAILY_FIELDS
    mkt_minute_fields = (set(field) & MKT_INTRADAY_FIELDS)
    mkt_fields = mkt_daily_fields | mkt_minute_fields
    if set(field) - set(mkt_fields):
        factors = translate_factors(
            list(set(field) - set(mkt_fields)), start_date.strftime('%Y%m%d'), **kwargs)
    else:
        factors = []
    fs_fields = list(set(field) & set(FS_FIELD))
    rest = set(field) - set(mkt_fields) - set(factors) - set(fs_fields)
    if rest:
        print (u"传入的参数fields中有 %s无法识别，请校验后重试。" % list(rest))

    lastest_stock_tdate = TODAY
    lastest_future_tdate = TODAY

    # parse end date
    # 用来智能判断今日行情是否到位，未到位则没有今天的这一行
    if end is None:
        data_due_date = os.getenv('data_due_date')
        if data_due_date:
            end_date = uniform_date(data_due_date)
        else:
            end_date = uniform_date(TODAY)
    else:
        end_date = uniform_date(end)
    if end_date is None:
        return

    if end_date >= uniform_date(TODAY) and get_trading_days(TODAY_STR, TODAY_STR):

        if stocks or funds or index:
            if freq in ['d', '1d']:
                lastest_stock_tdate = get_updated_date('daily_adj')
            elif freq in ['1m', '5m', '15m', '30m', '60m', 'm']:
                lastest_stock_tdate = get_updated_date('min_data')
        if futures:
            lastest_future_tdate = get_updated_date('future')
        if min([lastest_stock_tdate.strftime('%Y%m%d'), lastest_future_tdate.strftime('%Y%m%d')]) < TODAY_STR:
            print (u"今日行情还没有入库，默认获取截止到上一个交易日的数据。")
            end_date = uniform_date(YESTERDAY)

    if end_date < start_date:
        print (u"传入的start晚于end时间，请修正后重试。")
        return
    trading_days = [datetime.datetime.strptime(x, '%Y%m%d')
                    for x in get_td_from_cache(start_date.strftime('%Y%m%d'),
                                               end_date.strftime('%Y%m%d'))]
    if len(trading_days) == 0:
        print (u"传入的日期区间 start:%s, end:%s 不包含交易日，无数据返回" % (start, end))
        return
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')

    # validate symbol
    stock_fund_tdates_dict = AssetService.get_stock_fund_tdates(stocks + funds)
    filtered_stock_fund = []
    # 删除股票和基金中间不在交易区间的
    for univ in stocks + funds:
        if univ in stock_fund_tdates_dict:
            for one_tuple in stock_fund_tdates_dict[univ]:
                if str(one_tuple[0]) <= end_date_str and \
                        (one_tuple[1] is None or str(one_tuple[1]) > start_date_str):
                    filtered_stock_fund.append(univ)
                    break
    # 对于股票和基金没有就是无法交易
#         else:
#             filtered_stock_fund.append(univ)
    if len(set(stocks) - set(filtered_stock_fund)) > 0:
        print (u"传入的股票列表中：%s在给定的时间段内没有交易，数据不会出现在结果中。" %
               list(set(stocks) - set(filtered_stock_fund)))
    if len(set(funds) - set(filtered_stock_fund)) > 0:
        print (u"传入的基金列表中：%s在给定的时间段内没有交易，数据不会出现在结果中。" %
               list(set(funds) - set(filtered_stock_fund)))

    filtered_futures = []
    if futures:
        future_tdates_dict = AssetService.get_future_tdates(futures)
        # 删除futures中间不在交易区间的期货

        for future in futures:
            if future in future_tdates_dict:
                if str(future_tdates_dict[future][0]) < end_date_str and \
                        str(future_tdates_dict[future][1]) > start_date_str:
                    filtered_futures.append(future)
            else:
                # 比如连续合约暂时查不到
                filtered_futures.append(future)

        if len(set(futures) - set(filtered_futures)) > 0:
            print ("传入的期货列表中：%s在给定的时间段内没有交易，数据不会出现在结果中。" %
                   list(set(futures) - set(filtered_futures)))

    # 去掉不合法的symbol
    invalid_symbol = list(
        set(symbol) - (set(stocks) | set(futures) | set(index) | set(funds)))
    if invalid_symbol:
        print ("无法识别传入的symbol参数中的%s，请检查后重试。" % invalid_symbol)

    # 整理最后合法的symbol
    symbol = list(
        set(filtered_stock_fund) | set(filtered_futures) | set(index))
    if len(symbol) == 0:
        return pd.Panel()

    HelloUser({"symbol": symbol, 'field': field, 'start': start_date.strftime('%Y%m%d'),
               'end': end_date.strftime('%Y%m%d'), 'freq': freq,
               'style': style, 'adj': adj, 'uid': uid})
    if freq in ['d', '1d']:

        market_service = MarketService.create_with_service(symbol,
                                                           mkt_daily_fields,
                                                           mkt_minute_fields,
                                                           factors,
                                                           fs_fields,
                                                           adj=True if adj == 'pre' else False,
                                                           **kwargs)
        market_service.batch_load_daily_data(trading_days)
        data = market_service.slice(symbols=symbol, fields=field,
                                    end_date=end_date, freq='d', start_date=start_date,
                                    style=style, rtype='frame', time_range=None)
        HelloUser({"symbol": symbol, 'field': field, 'start': start_date.strftime('%Y%m%d'),
                   'end': end_date.strftime('%Y%m%d'), 'freq': freq,
                   'style': style, 'adj': adj, 'uid': uid, 'cost': time.time() - entry_time})
        return pd.Panel(data).replace([None], np.nan)
    if freq in ['1m', '5m', '15m', '30m', '60m', 'm']:
        if 'IFZ0' in symbol:
            print ("IFZ0暂不支持分钟线数据。")
            symbol.remove('IFZ0')

        mkt_daily_fields = list(set(mkt_daily_fields) - set(mkt_minute_fields))

        if mkt_daily_fields:
            print ("传入字段：%s为日线字段，暂时没有分钟相关的数据。" % mkt_daily_fields)
        market_service = MarketService.create_with_service(symbol,
                                                           [],
                                                           mkt_minute_fields,
                                                           factors,
                                                           fs_fields,
                                                           adj=True if adj == 'pre' else False,
                                                           **kwargs)
        market_service.batch_load_daily_data(trading_days)
        market_service.rolling_load_minute_data(trading_days, None, freq)
        tick_roller = TickRoller(market_service)
        data_min = tick_roller.slice(prepare_dates=trading_days, end_time=MAX_END_DATE,
                                     time_range=MAX_MINUTE_LENGTH,
                                     fields=field, symbols=symbol, style=style,
                                     rtype='frame')
        HelloUser({"symbol": symbol, 'field': field, 'start': start_date.strftime('%Y%m%d'),
                   'end': end_date.strftime('%Y%m%d'), 'freq': freq,
                   'style': style, 'adj': adj, 'uid': uid, 'cost': time.time() - entry_time})

        return pd.Panel(data_min).replace([None], np.nan)


def __format_str_to_arr(param, param_name):
    if isinstance(param, basestring):
        if ',' in param:
            param = param.split(',')
        elif ';' in param:
            param = param.split(';')
        else:
            param = [param]
    elif isinstance(param, list):
        for item in param:
            if not isinstance(item, basestring):
                print (u"参数%s的值支持list of str或者str类型，请校验后重新输入。" % param_name)
                return []
    return [item.strip() for item in param if isinstance(item, basestring)]
