# 克隆自聚宽文章：https://www.joinquant.com/post/16126
# 标题：商品期货多因子 全市场对冲模型
# 作者：test180208

# 期货日频多品种，横截面多因子模型
# 建议给予1000000元，2012年1月1日至今回测
# 导入函数库
# 3因子定稿

from jqdata import *
import talib
from math import isnan
import re
from jqfactor import get_factor_values
from jqfactor import standardlize
from jqfactor import winsorize_med
from six import StringIO, BytesIO


def initialize(context):
    # 设置参数
    set_parameter(context)
    # 价格列表初始化
    set_future_list(context)
    # 导入写入CSV文件
    csv_setting(context)
    # 设定基准银华日利，在多品种的回测当中基准没有参考意义
    set_benchmark('511880.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    ### 期货相关设定 ###
    # 设定账户为金融账户
    set_subportfolios([SubPortfolioConfig(cash=context.portfolio.starting_cash, type='futures')])
    # 期货类每笔交易时的手续费是：买入时万分之1,卖出时万分之1,平今仓为万分之1
    set_order_cost(OrderCost(open_commission=0.0001, close_commission=0.0001, close_today_commission=0.0001),
                   type='futures')
    # 设定保证金比例
    set_option('futures_margin_rate', 0.15)
    # 设置滑点（单边万5，双边千1）
    set_slippage(PriceRelatedSlippage(0.001), type='future')
    # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security=get_future_code('RB'))
    # 开盘时运行
    run_weekly(market_open, 1, time='open', reference_security=get_future_code('RB'))
    # 交易运行
    run_weekly(Trade, 1, time='open', reference_security=get_future_code('RB'))
    # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security=get_future_code('RB'))


# 参数设置函数
def set_parameter(context):
    #######变量设置########
    g.domMonth = {
        'MA': ['01', '05', '09'],
        'IC': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'IF': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'IH': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'TF': ['03', '06', '09', '12'],
        'T': ['03', '06', '09', '12'],
        'CU': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'AL': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'ZN': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'PB': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'NI': ['01', '05', '09'],
        'SN': ['01', '05', '09'],
        'AU': ['06', '12'],
        'AG': ['06', '12'],
        'RB': ['01', '05', '10'],
        'HC': ['01', '05', '10'],
        'BU': ['06', '09', '12'],
        'RU': ['01', '05', '09'],
        'M': ['01', '05', '09'],
        'Y': ['01', '05', '09'],
        'A': ['01', '05', '09'],
        'P': ['01', '05', '09'],
        'C': ['01', '05', '09'],
        'CS': ['01', '05', '09'],
        'JD': ['01', '05', '09'],
        'L': ['01', '05', '09'],
        'V': ['01', '05', '09'],
        'PP': ['01', '05', '09'],
        'J': ['01', '05', '09'],
        'JM': ['01', '05', '09'],
        'I': ['01', '05', '09'],
        'SR': ['01', '05', '09'],
        'CF': ['01', '05', '09'],
        'ZC': ['01', '05', '09'],
        'FG': ['01', '05', '09'],
        'TA': ['01', '05', '09'],
        'MA': ['01', '05', '09'],
        'OI': ['01', '05', '09'],
        'RM': ['01', '05', '09'],
        'SF': ['01', '05', '09'],
        'SM': ['01', '05', '09'],
        'AP': ['01', '05', '10'],
    }

    g.LastRealPrice = {}  # 最新真实合约价格字典(用于吊灯止损）
    g.HighPrice = {}  # 各品种最高价字典（用于吊灯止损）
    g.LowPrice = {}  # 各品种最低价字典（用于吊灯止损）
    g.future_list = []  # 设置期货品种列表
    g.TradeLots = {}  # 各品种的交易手数信息
    g.PriceArray = {}  # 信号计算价格字典
    g.Price_dict = {}  # 各品种价格列表字典
    g.MappingReal = {}  # 真实合约映射（key为symbol，value为主力合约）
    g.MappingIndex = {}  # 指数合约映射 （key为 symbol，value为指数合约
    g.StatusTimer = {}  # 当前状态计数器
    g.ATR = {}
    g.CurrentPrice = 0
    g.Price_DaysAgo = 0
    g.Momentum = {}
    g.ClosePrice = {}
    g.ILLIQ = {}
    g.MarginRate = 0.1
    g.Score = {}
    g.MappingNext = {}  # 映射合约
    g.NextPrice = {}  # 远期价格序列
    g.RealPrice = {}  # 主力价格序列
    g.RollYield = {}  # 展期收益率
    g.Volume = {}  # 成交量序列
    g.VSTD_Volume = {}  # 成交量变异系数
    g.SkewPrice = {}  # 偏度价格序列
    g.Skew = {}  # 偏度
    g.Basis = {}  # 基本面因子：基差
    g.Hycc = {}  # 基本面因子：会员持仓

    #######参数设置########
    g.ATRWindow = 20  # ATR回溯窗口长度
    g.MomentumWindow = 30  # 截面动量长度
    g.ILLIQWindow = 10  # 流动性因子长度
    g.Range = 0.3  # 做多做空名单的头尾长度
    g.NATR_stop = 2  # 追踪止损长度
    # 交易的期货品种信息
    g.instruments = ['AL', 'NI', 'CU', 'PB', 'AG',
                     'RU', 'MA', 'PP', 'TA', 'L', 'V',
                     'M', 'P', 'Y', 'OI', 'C', 'CS', 'JD', 'SR',
                     'HC', 'J', 'I', 'SF', 'RB', 'ZC', 'FG']


# 导入读入csv文件
def csv_setting(context):
    # 基差setting
    body1 = read_file("Basis_xzh1.csv")
    df1 = pd.read_csv(BytesIO(body1))
    df1.fillna(0)
    df1.index = df1['Unnamed: 0']
    del df1['Unnamed: 0']
    g.df1 = df1[g.instruments]

    # 会员持仓setting
    body2 = read_file("hycc_TS3.csv")
    df2 = pd.read_csv(BytesIO(body2))
    df2.fillna(0)
    df2.index = df2['Unnamed: 0']
    del df2['Unnamed: 0']
    g.df2 = df2[g.instruments]


# 价格列表初始化
def set_future_list(context):
    for ins in g.instruments:
        idx = get_future_code(ins)
        dom = get_dominant_future(ins)
        # 填充映射字典
        g.MappingIndex[ins] = idx
        g.MappingReal[ins] = dom
        g.StatusTimer[ins] = 0
        # 设置主力合约已上市的品种基本参数
        if dom == '':
            pass
        else:
            if dom not in g.future_list:
                g.future_list.append(dom)
                g.HighPrice[dom] = False
                g.LowPrice[dom] = False


'''
换月模块逻辑（ins是期货品种的symbol（如‘RB’），dom或future指合约（如'RB1610.XSGE'）,idx指指数合约（如’RB8888.XSGE‘）
    1.在第一天开始时，将所有期货品种最初的主力合约写入MappingReal与MappingIndex当中
    2.每天开盘获取一遍ins对应的主力合约，判断是否在MappingReal中，若不在，则执行replace模块
    3.replace模块中，卖出原来持有的主力合约，等量买入新合约；修改MappingReal
'''


## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：' + str(context.current_dt.time()))
    send_message('开始交易')

    # 过滤无主力合约的品种，传入并修改期货字典信息
    for ins in g.instruments:
        dom = get_dominant_future(ins)
        if dom == '':
            pass
        else:
            # 判断是否执行replace_old_futures
            if dom == g.MappingReal[ins]:
                pass
            else:
                replace_old_futures(context, ins, dom)
                g.future_list.append(dom)
                g.HighPrice[dom] = False
                g.LowPrice[dom] = False

            g.TradeLots[dom] = get_lots(context.portfolio.starting_cash / 10, ins)


## 开盘时运行函数
def market_open(context):
    # 输出函数运行时间
    # log.info('函数运行时间(market_open):'+str(context.current_dt.time()))

    # 以下是主循环
    for ins in g.instruments:
        get_nextDom(context, ins)
        # 过滤空主力合约品种
        if g.MappingReal[ins] != '':
            IndexFuture = g.MappingIndex[ins]
            RealFuture = g.MappingReal[ins]
            NextFuture = g.MappingNext[ins]
            # 获取当月合约交割日期
            end_date = get_CCFX_end_date(RealFuture)
            # 当月合约交割日当天不开仓
            if (context.current_dt.date() == end_date):
                return
            else:
                g.LastRealPrice[RealFuture] = attribute_history(RealFuture, 1, '1d', ['close'])['close'][-1]
                # 获取价格
                g.PriceArray[IndexFuture] = attribute_history(IndexFuture, max(g.MomentumWindow,
                                                                               max(g.ILLIQWindow, g.ATRWindow)) + 5,
                                                              '1d', ['close', 'open', 'high', 'low'])
                g.NextPrice[NextFuture] = attribute_history(NextFuture, 1, '1d', ['close'])
                g.RealPrice[RealFuture] = attribute_history(RealFuture, 1, '1d', ['close'])

                # 如果没有数据，返回
                if len(g.PriceArray[IndexFuture]) < 1:
                    return
                else:
                    # 获取ATR
                    g.close = np.array(g.PriceArray[IndexFuture]['close'])
                    g.high = np.array(g.PriceArray[IndexFuture]['high'])
                    g.low = np.array(g.PriceArray[IndexFuture]['low'])
                    g.ATR[IndexFuture] = talib.ATR(g.high, g.low, g.close, g.ATRWindow)[-1]

                    # 截面动量计算
                    g.ClosePrice[ins] = g.PriceArray[IndexFuture]['close']
                    g.CurrentPrice = g.ClosePrice[ins][-1]
                    g.Price_DaysAgo = g.ClosePrice[ins][-g.MomentumWindow]
                    g.Momentum[ins] = g.CurrentPrice / g.Price_DaysAgo

                    #  流动性因子计算
                    FP = get_extras('futures_positions', IndexFuture, end_date=context.current_dt, df=True,
                                    count=max(g.MomentumWindow, max(g.ILLIQWindow, g.ATRWindow)))
                    ILLIQ = 0
                    for i in range(g.ILLIQWindow):
                        ILLIQ += abs(1 - g.close[-i - 1] / g.close[-i - 2]) / FP[IndexFuture][-i - 1]
                    g.ILLIQ[ins] = -ILLIQ

                    # 展期收益率计算
                    g.PriceNext = g.NextPrice[NextFuture]['close'][-1]
                    g.PriceCurrent = g.RealPrice[RealFuture]['close'][-1]

                    NextDate = re.findall(r"\d+\d*", NextFuture)
                    RealDate = re.findall(r"\d+\d*", RealFuture)

                    if NextDate[0][:2] > RealDate[0][:2]:
                        g.DM = int(NextDate[0][-2:]) + 12 - int(RealDate[0][-2:])
                    else:
                        g.DM = int(NextDate[0][-2:]) - int(RealDate[0][-2:])

                    # 展期收益率(负号代表因子值大的sell）
                    g.RollYield[ins] = (g.PriceCurrent - g.PriceNext) / g.PriceNext / g.DM

                    # 归一化VSTD计算（可不断增加新因子）
                    g.Volume[ins] = attribute_history(IndexFuture, 180, '1d', ['volume'])['volume']
                    try:
                        g.VSTD_Volume[ins] = g.Volume[ins].std() / g.Volume[ins].mean()
                    except:
                        pass

                    # 偏度计算（可不断增加新因子）
                    g.SkewPrice[ins] = attribute_history(IndexFuture, 180, '1d', ['close'])['close']
                    g.Skew[ins] = get_Skew(g.SkewPrice[ins])

                    # 基差计算
                    sort_basis(context)

                    # 会员持仓计算
                    sort_hycc(context)


## 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):' + str(context.current_dt.time())))
    # 得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：' + str(_trade))
    log.info('一天结束')
    log.info('##############################################################')


## 交易模块
def Trade(context):
    # 获取交易名单
    BuyList, SellList = rank_futures(context)

    for ins in g.instruments:
        RealFuture = g.MappingReal[ins]
        IndexFuture = g.MappingIndex[ins]
        if RealFuture in g.LastRealPrice.keys():
            if ins in BuyList and context.portfolio.long_positions[RealFuture].total_amount == 0:
                if context.portfolio.long_positions[RealFuture].total_amount != 0:
                    log.info('空头有持仓：%s' % (RealFuture))
                order_target(RealFuture, 0, side='short')
                order_target(RealFuture, g.TradeLots[RealFuture], side='long')
                g.HighPrice[RealFuture] = g.LastRealPrice[RealFuture]
                g.LowPrice[RealFuture] = False
                log.info('正常买多合约：%s' % (RealFuture))

            elif ins in SellList and context.portfolio.short_positions[RealFuture].total_amount == 0:
                if context.portfolio.short_positions[RealFuture].total_amount != 0:
                    log.info('多头有持仓：%s' % (RealFuture))
                order_target(RealFuture, 0, side='long')
                order_target(RealFuture, g.TradeLots[RealFuture], side='short')
                g.LowPrice[RealFuture] = g.LastRealPrice[RealFuture]
                g.HighPrice[RealFuture] = False
                log.info('正常卖空合约：%s' % (RealFuture))
            elif ins not in BuyList and RealFuture in context.portfolio.long_positions.keys():
                order_target(RealFuture, 0, side='long')
            elif ins not in SellList and RealFuture in context.portfolio.short_positions.keys():
                order_target(RealFuture, 0, side='short')
            else:
                TrailingStop(context, RealFuture, IndexFuture)


# 追踪止损模块（ATR止损）
def TrailingStop(context, RealFuture, IndexFuture):
    # 记录多空仓位
    long_positions = context.portfolio.long_positions
    short_positions = context.portfolio.short_positions

    # 通过for循环逐一平仓（多头）
    if RealFuture in long_positions.keys():
        if long_positions[RealFuture].total_amount > 0:
            if g.HighPrice[RealFuture]:
                g.HighPrice[RealFuture] = max(g.HighPrice[RealFuture], g.LastRealPrice[RealFuture])
                if g.LastRealPrice[RealFuture] < g.HighPrice[RealFuture] - g.NATR_stop * g.ATR[IndexFuture]:
                    log.info('多头止损:\t' + RealFuture)
                    order_target(RealFuture, 0, side='long')

    # 通过for循环逐一平仓（空头）
    if RealFuture in short_positions.keys():
        if short_positions[RealFuture].total_amount > 0:
            if g.LowPrice[RealFuture]:
                g.LowPrice[RealFuture] = min(g.LowPrice[RealFuture], g.LastRealPrice[RealFuture])
                if g.LastRealPrice[RealFuture] > g.LowPrice[RealFuture] + g.NATR_stop * g.ATR[IndexFuture]:
                    log.info('空头止损:\t' + RealFuture)
                    order_target(RealFuture, 0, side='short')


# 获得当前远月合约
def get_nextDom(context, ins):
    # 获取当月主力合约
    dom = get_dominant_future(ins)
    if dom != '':
        # 获取合约交割月份时间，如['1601']
        YD = re.findall(r"\d+\d*", dom)
        try:
            # 获取远月合约——读取“历史上的每个合约主力月份”，读取月份后两位，往后移一位
            nextDomMonth = g.domMonth[ins][g.domMonth[ins].index(YD[0][-2:]) + 1]
            # 替换远月合约日期，合成远月合约名，如['RB1601.XSGE']
            g.MappingNext[ins] = dom.replace(YD[0][-2:], nextDomMonth)
        except:
            # 否则年份+1，月份取第一位
            if YD[:2] != YD[-2:]:
                nextDomMonth = g.domMonth[ins][0]
                nextDom = dom.replace(YD[0][:2], str(int(YD[0][:2]) + 1))
                g.MappingNext[ins] = nextDom.replace(YD[0][-2:], g.domMonth[ins][0])
            else:
                # 防止出现1212这样的时间
                NewYD = str(int(YD[0]) + 100).replace(YD[0][-2:], g.domMonth[ins][0])
                g.MappingNext[ins] = dom.replace(YD[0], NewYD)
    else:
        pass


# 移仓模块：当主力合约更换时，平当前持仓，更换为最新主力合约
def replace_old_futures(context, ins, dom):
    LastFuture = g.MappingReal[ins]

    if LastFuture in context.portfolio.long_positions.keys():
        lots_long = context.portfolio.long_positions[LastFuture].total_amount
        order_target(LastFuture, 0, side='long')
        order_target(dom, lots_long, side='long')
        print('主力合约更换，平多仓换新仓')

    if LastFuture in context.portfolio.short_positions.keys():
        lots_short = context.portfolio.short_positions[dom].total_amount
        order_target(LastFuture, 0, side='short')
        order_target(dom, lots_short, side='short')
        print('主力合约更换，平空仓换新仓')

    g.MappingReal[ins] = dom


# 多因子排序
def rank_futures(context):
    future_list = []
    for ins in g.instruments:
        future = get_dominant_future(ins)
        future_list.append(future)

    # 生成多列Rank值排序
    statements = pd.DataFrame([g.Momentum, g.ILLIQ, g.RollYield, g.VSTD_Volume, g.Skew, g.Basis, g.Hycc]).T
    # statements=pd.DataFrame([g.ILLIQ,g.Momentum,g.RollYield]).T

    # nark化列名称和值
    statements.columns = ['Momentum', 'ILLIQ', 'RollYield', 'VSTD_Volume', 'Skew', 'Basis', 'Hycc']
    # statements.columns = ['ILLIQ','Momentum','RollYield']
    statements['Momentum_rank'] = statements['Momentum'].rank()
    statements['ILLIQ_rank'] = statements['ILLIQ'].rank()
    statements['RollYield_rank'] = statements['RollYield'].rank()
    statements['VSTD_rank'] = statements['VSTD_Volume'].rank()
    statements['Skew_rank'] = statements['Skew'].rank()
    statements['Basis_rank'] = statements['Basis'].rank()
    statements['Hycc_rank'] = statements['Hycc'].rank()

    '''
    # 标准化后加权打分（方法1）先剔除3MAD异常值，再标准化
    statements = standardlize(statements, inf2nan=True, axis=0)
    statements = statements.fillna(0)
    data_final = 0.33*statements['Momentum'] \
                + 0*statements['ILLIQ'] \
                + 0.33*statements['RollYield'] \
                - 0.33*statements['Basis']
    '''
    # 因子权重和因子值rank相乘（方法2，直接rank排序）
    data_final = 1 * statements['Momentum_rank'] \
                 + 1 * statements['ILLIQ_rank'] \
                 + 1 * statements['RollYield_rank'] \
                 - 1 * statements['VSTD_rank'] \
                 - 1 * statements['Skew_rank'] \
                 - 1 * statements['Basis_rank'] \
                 - 0 * statements['Hycc_rank']

    print(statements)

    data_final.sort_index(ascending=False)

    stocklist_head = list(data_final.index)[:int(len(list(data_final.index)) * g.Range)]
    stocklist_tail = list(data_final.index)[-int(len(list(data_final.index)) * g.Range) - 1:]

    return stocklist_head, stocklist_tail


# 得到期货现货基差因子
def sort_basis(context):
    ## 现在的自然日日期
    date = context.current_dt.strftime('%Y-%m-%d')
    ## 此日期的各合约展期
    ## 修改这行可以将基差改成基差均值
    jc_today_df1 = g.df1.loc[date, :].dropna(how='any')
    ## 选出多品种列表、空品种列表
    for ins in jc_today_df1.keys():
        g.Basis[ins] = jc_today_df1[ins]


# 得到会员持仓因子
def sort_hycc(context):
    ## 现在的自然日日期
    date = context.current_dt.strftime('%Y-%m-%d')
    ## 此日期的各合约展期
    ## 修改这行可以将基差改成基差均值
    jc_today_df2 = g.df2.loc[date, :].dropna(how='any')
    ## 选出多品种列表、空品种列表
    for ins in jc_today_df2.keys():
        g.Hycc[ins] = jc_today_df2[ins]


# 得到偏度
def get_Skew(data_):
    n = len(data_)  # 10000个数
    niu = 0.0  # niu表示平均值,即期望.
    niu2 = 0.0  # niu2表示平方的平均值
    niu3 = 0.0  # niu3表示三次方的平均值
    for a in data_:
        niu += a
        niu2 += a ** 2
        niu3 += a ** 3
    niu /= n
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu * niu)
    skew = (niu3 - 3 * niu * sigma ** 2 - niu ** 3) / (sigma ** 3)
    return skew


'''增加删除过去合约'''


# 获取当天时间正在交易的期货主力合约函数
def get_future_code(symbol):
    future_code_list = {'A': 'A8888.XDCE', 'AG': 'AG8888.XSGE', 'AL': 'AL8888.XSGE', 'AU': 'AU8888.XSGE',
                        'B': 'B8888.XDCE', 'BB': 'BB8888.XDCE', 'BU': 'BU8888.XSGE', 'C': 'C8888.XDCE',
                        'CF': 'CF8888.XZCE', 'CS': 'CS8888.XDCE', 'CU': 'CU8888.XSGE', 'ER': 'ER8888.XZCE',
                        'FB': 'FB8888.XDCE', 'FG': 'FG8888.XZCE', 'FU': 'FU8888.XSGE', 'GN': 'GN8888.XZCE',
                        'HC': 'HC8888.XSGE', 'I': 'I8888.XDCE', 'IC': 'IC8888.CCFX', 'IF': 'IF8888.CCFX',
                        'IH': 'IH8888.CCFX', 'J': 'J8888.XDCE', 'JD': 'JD8888.XDCE', 'JM': 'JM8888.XDCE',
                        'JR': 'JR8888.XZCE', 'L': 'L8888.XDCE', 'LR': 'LR8888.XZCE', 'M': 'M8888.XDCE',
                        'MA': 'MA8888.XZCE', 'ME': 'ME8888.XZCE', 'NI': 'NI8888.XSGE', 'OI': 'OI8888.XZCE',
                        'P': 'P8888.XDCE', 'PB': 'PB8888.XSGE', 'PM': 'PM8888.XZCE', 'PP': 'PP8888.XDCE',
                        'RB': 'RB8888.XSGE', 'RI': 'RI8888.XZCE', 'RM': 'RM8888.XZCE', 'RO': 'RO8888.XZCE',
                        'RS': 'RS8888.XZCE', 'RU': 'RU8888.XSGE', 'SF': 'SF8888.XZCE', 'SM': 'SM8888.XZCE',
                        'SN': 'SN8888.XSGE', 'SR': 'SR8888.XZCE', 'T': 'T8888.CCFX', 'TA': 'TA8888.XZCE',
                        'TC': 'TC8888.XZCE', 'TF': 'TF8888.CCFX', 'V': 'V8888.XDCE', 'WH': 'WH8888.XZCE',
                        'WR': 'WR8888.XSGE', 'WS': 'WS8888.XZCE', 'WT': 'WT8888.XZCE', 'Y': 'Y8888.XDCE',
                        'ZC': 'ZC8888.XZCE', 'ZN': 'ZN8888.XSGE', 'AP': 'AP8888.XZCE'}
    try:
        return future_code_list[symbol]
    except:
        return 'WARNING: 无此合约'


'''
# # 获取交易手数函数(无ATR版本）
def get_lots(cash,symbol):
    # 获取合约规模(Contract Size)，也称交易单位
    future_Contract_Size = {'A':10, 'AG':15, 'AL':5, 'AU':1000,
                        'B':10, 'BB':500, 'BU':10, 'C':10,
                        'CF':5, 'CS':10, 'CU':5, 'ER':10,
                        'FB':500, 'FG':20, 'FU':50, 'GN':10,
                        'HC':10, 'I':100, 'IC':200, 'IF':300,
                        'IH':300, 'J':100, 'JD':5, 'JM':60,
                        'JR':20, 'L':5, 'LR':10, 'M':10,
                        'MA':10, 'ME':10, 'NI':1, 'OI':10,
                        'P':10, 'PB':5, 'PM':50, 'PP':5,
                        'RB':10, 'RI':20, 'RM':10, 'RO':10,
                        'RS':10, 'RU':10, 'SF':5, 'SM':5,
                        'SN':1, 'SR':10, 'T':10000, 'TA':5,
                        'TC':100, 'TF':10000, 'V':5, 'WH':20,
                        'WR':10, 'WS':50, 'WT':10, 'Y':10,
                        'ZC':100, 'ZN':5,'AP':10}
    future = get_dominant_future(symbol)
    # 获取价格list
    Price_dict = attribute_history(future,10,'1d',['open'])
    # 如果没有数据，返回
    if len(Price_dict) == 0:
        return
    else:
        # 获得最新开盘价，计算能够下单多少手
        open_price = Price_dict.iloc[-1]
    # 返回手数（价格*合约规模=名义价值）
    # 保证金使用，控制在33%
    # 合约保证金的表达式是：open_price*future_Contract_Size[symbol]*g.MarginRate
    return cash*0.33/(open_price*future_Contract_Size[symbol]*g.MarginRate)
'''


# 获取交易手数函数（ATR倒数头寸）
def get_lots(cash, symbol):
    future_coef_list = {'A': 10, 'AG': 15, 'AL': 5, 'AU': 1000,
                        'B': 10, 'BB': 500, 'BU': 10, 'C': 10,
                        'CF': 5, 'CS': 10, 'CU': 5, 'ER': 10,
                        'FB': 500, 'FG': 20, 'FU': 50, 'GN': 10,
                        'HC': 10, 'I': 100, 'IC': 200, 'IF': 300,
                        'IH': 300, 'J': 100, 'JD': 5, 'JM': 60,
                        'JR': 20, 'L': 5, 'LR': 10, 'M': 10,
                        'MA': 10, 'ME': 10, 'NI': 1, 'OI': 10,
                        'P': 10, 'PB': 5, 'PM': 50, 'PP': 5,
                        'RB': 10, 'RI': 20, 'RM': 10, 'RO': 10,
                        'RS': 10, 'RU': 10, 'SF': 5, 'SM': 5,
                        'SN': 1, 'SR': 10, 'T': 10000, 'TA': 5,
                        'TC': 100, 'TF': 10000, 'V': 5, 'WH': 20,
                        'WR': 10, 'WS': 50, 'WT': 10, 'Y': 10,
                        'ZC': 100, 'ZN': 5, 'AP': 10}
    RealFuture = get_dominant_future(symbol)
    IndexFuture = get_future_code(symbol)
    # 获取价格list
    Price_dict = attribute_history(IndexFuture, 10, '1d', ['open'])
    # 如果没有数据，返回
    if len(Price_dict) == 0:
        return
    else:
        open_future = Price_dict.iloc[-1]
    # 返回手数
    if IndexFuture in g.ATR.keys():
        # 每次使用5%资金开仓交易
        # 合约价值的表达式是：g.ATR[IndexFuture]*future_coef_list[symbol]
        return cash * 0.05 / (g.ATR[IndexFuture] * future_coef_list[symbol])
    else:  # 函数运行之初会出现没将future写入ATR字典当中的情况
        return 0


# 获取金融期货合约到期日
def get_CCFX_end_date(fature_code):
    # 获取金融期货合约到期日
    return get_security_info(fature_code).end_date