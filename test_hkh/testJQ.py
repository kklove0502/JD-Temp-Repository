import jqdatasdk as jqd
jqd.auth('18018760868','760868')

print(jqd.get_query_count())

df = jqd.get_price(security, start_date=None, end_date=None, frequency='daily', fields=None, skip_paused=False, fq='pre', count=None)
df = jqd.get_bars('600519.XSHG', 10, unit='1d',fields=['date','open','high','low','close'],include_now=False,end_dt='2018-12-05')