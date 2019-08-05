# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import inspect
import locale
import os
import time
from copy import deepcopy
import traceback

import sys
import requests
from requests.exceptions import ReadTimeout
from . import settings, global_cache_map

from ..uqer import session
from ..config import *
from ..version import __version__

from platform import python_version
import json
import codecs

csv_encoding = 'gb2312'
request_encoding = 'UTF-8'

timeout = 60
retry_interval = 2
max_retries = 5

client_info = json.dumps({"python_version": python_version(),
"client_version": __version__,
"module":"uqer_sdk"})

def get_real_string(input_string):
    if sys.version_info.major == 2:
        out = input_string.encode(locale.getpreferredencoding())
    else:
        out = input_string
    return out


def get_http_result(httpClient, requestString, gw, max_retries=max_retries):
 
    for i in range(1, max_retries + 1):
        try:
            source = 'mercury_sdk:' + os.environ.get('DatayesPrincipalName', 'invalid_user')
            result = session.get("https://%s:%d/data/v1%s" % (httpClient[2], httpClient[3], requestString),
                                  headers={'Connection': 'close', 
                                           "Authorization": "Bearer " + get_token(), 
                                           'SOURCE': source,
                                           'CLIENT_INFO': client_info},
                                  timeout=timeout)
            return result

        except Exception as e:
            time.sleep(retry_interval)


def __getCSV__(requestString, httpClient, gw=True):
    try:
        result = get_http_result(httpClient, requestString, gw)
        if result.status_code == 400:
            raise Exception('请检查输入参数，可能某列表输入参数过长')
        if int(result.headers.get('dyes-rsp-count', 0)) == 100000:
            result2 = get_http_result(httpClient, requestString + '&pagenum=2', gw)
            if int(result2.headers.get('dyes-rsp-count', 0)) > 0:
                raise Exception(get_real_string('Data API查询返回结果超过数据记录上限（十万条记录），请修改查询条件分步查询'))
        return result.text
    except ReadTimeout:
        raise Exception('查询服务超时')
    except Exception as e:
        raise e

def get_cache_key(frame):
    args, _, _, values = inspect.getargvalues(frame)
    func_name = inspect.getframeinfo(frame)[2]
    cache_key = hash([values[arg] for arg in args].__str__())
    return func_name, cache_key

def get_data_from_cache(func_name, cache_key):
    return

def put_data_in_cache(func_name, cache_key, data):
    return

def splist(l, s):
    return [l[i:i+s] for i in range(len(l)) if i %s == 0]

def is_no_data_warn(csvString, print_msg):
    if csvString.startswith('-1:No Data Returned'):
        if print_msg:
            print('没有数据返回。请检查输入参数，若仍有问题，可联系service.uqer@datayes.com')
        return True
    return False

def handle_error(csvString, api_name):
    if csvString.startswith('-403:Need Privilege'):
        result = '无%s接口使用权限，您可以购买优矿专业版（https://uqer.io/pro） 或联系 4000 820 386 购买数据' % api_name
    elif csvString.startswith('-403:Need login'):
        result = '您未登陆'
    elif csvString.startswith('-2:Invalid Request Parameter'):
        result = '无效的请求参数。请检查输入参数，若仍有问题，可联系service.uqer@datayes.com'
    elif csvString.startswith('-3:Service Suspend'):
        result = '服务终止。请联系service.uqer@datayes.com'
    elif csvString.startswith('-4:Internal Server Error'):
        result = '内部服务器错误。请联系service.uqer@datayes.com'
    elif csvString.startswith('-5:Server Busy'):
        result = '服务器拥堵。可能是海量用户在同一时间集中调用该数据造成，可稍后再次尝试。' \
                 '如长时间未改善，或频繁出现该问题，可联系sevice.uqer@datayes.com'
    elif csvString.startswith('-6:Trial Times Over'):
        result = '试用次数达到限制。您对该数据的试用权限已经到期'
    elif csvString.startswith('-7:Query Timeout'):
        result = '请求超时。可能您请求的数据量较大或服务器当前忙'
    elif csvString.startswith('-8:Query Failed'):
        result = '请求失败，请联系service.uqer@datayes.com'
    elif csvString.startswith('-9:Required Parameter Missing'):
        result = '必填参数缺失。请仔细复核代码，将其中的参数补充完整后再次尝试'
    elif csvString.startswith('-11:The number of API calls reached limit'):
        result = '当日调用次数达到上限，请优化代码调用逻辑。每日0点重新计数'
    else:
        result = csvString
    err_msg = result
    print (err_msg)

    raise Exception(get_real_string(err_msg))

def get_token():
    import os
    ACCESS_TOKEN = "cloud_sso_token"
    DataAPI_TOKEN = "access_token"

    if os.environ.get(DataAPI_TOKEN):
        return os.environ[DataAPI_TOKEN]

    if os.environ.get(ACCESS_TOKEN):
        access_token = os.environ[ACCESS_TOKEN]
        r2 = None
        try:
            r2 = session.post(TOKEN_URL,
                               data={'grant_type': 'permanent'})
            token = r2.json()['access_token']
            os.environ[DataAPI_TOKEN] = token

            return token
        except:
            print ('获取通联数据权限凭证出错，可能是您的token无效或者token过期，请重新登录\n')

            return ''
    else:
        print ('当前环境下没有 token，可能是您未登录，请先登录哦')
        return ''

def __getConn__():
    return server

def __formatDate__(inputDate):
    return inputDate

def lowcase_keys(d):
    result = {}
    for key, value in d.items():
        lower_key = key.lower()
        result[lower_key] = value
    return result

def is_pro_user():
    return True


def showtraceback(self, exc_tuple=None, filename=None, tb_offset=None,
                  exception_only=False):
    import traceback
    import sys

    etype, value, tb = self._get_exc_info(exc_tuple)
    listing = traceback.format_exception(etype, value, tb)
    last_message = ''
    lineno = None
    text = ''

    if listing:
        last_message = listing[-1].decode('utf-8')
    
    except_msg = 'Exception:'
    if last_message.startswith(except_msg):
        last_message = '异常:' + last_message[len(except_msg):]

    for filename, lineno, module, text in traceback.extract_tb(tb):
        if filename.startswith('<mercury-input') and \
           not text.startswith('display=True, return_quartz_data=True'):
            break

    if lineno:
        print('行号: %s\n代码: %s\n%s' %(lineno, text, last_message))
    else:
        print(last_message)


def pretty_traceback():
    if not settings.pretty_traceback_enabled:
        return
    import os
    if os.environ.get('env') in ['qa', 'stg', 'prd']:
        try:
            import IPython
            IPython.core.interactiveshell.InteractiveShell.showtraceback = showtraceback
        except:
            pass


from .version import __version__ as dataapi_version

version_url = '%s/w/api/info'%(SDK_URL)
files_url = '%s/w/api/download'%(SDK_URL)


def get_api_version():
    res = requests.get(version_url).json()
    if res.get('code') != 200:
        return '', []

    remote_version = res['data']['version']
    py_files = res['data']['py_files']

    return remote_version, py_files


def get_api_file(filename):
    code = 0
    text = ''

    for i in range(3):
        try:
            response = requests.get('%s/%s'%(files_url, filename))
            code, text = response.status_code, response.text
            if code == 200:
                break
        except:
            pass
        import time;time.sleep(1)
    
    return code, text

def replace_api_files():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    remote_version, py_files = get_api_version()
    if not remote_version or remote_version <= dataapi_version:
        return

    for filename in py_files:
        if filename == '__init__':
            continue

        filename = '%s.py'%(filename)
        file_path = os.path.join(dir_path, filename)

        status_code, response = get_api_file(filename)

        if status_code!=200:
            # 文件可能是没下载正确，就不更新了
            continue

        if os.path.isfile(file_path):
            os.remove(file_path)

        with codecs.open(file_path, 'w', "utf-8-sig") as f:
            f.write(response)

    version_str = "__version__ = '%s'"%(remote_version)
    version_file_path = os.path.join(dir_path, 'version.py')
    with open(version_file_path, 'w') as f:
        f.write(version_str)
        print('UQER SDK的DataAPI模块版本由%s升级到%s'%(dataapi_version, remote_version))