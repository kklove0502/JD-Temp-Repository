import QUANTAXIS as QA

Account = QA.QA_Account()

B = QA.QA_BacktestBroker()

Order = Account.send_order(code='000001',
                           price=11,
                           money=Account.cash_available,
                           time='2018-05-09',
                           towards=QA.ORDER_DIRECTION.BUY,
                           order_model=QA.ORDER_MODEL.MARKET,
                           amount_model=QA.AMOUNT_MODEL.BY_MONEY
                           )

print('ORDER的占用资金: {}'.format((Order.amount*Order.price)*(1+Account.commission_coeff)))