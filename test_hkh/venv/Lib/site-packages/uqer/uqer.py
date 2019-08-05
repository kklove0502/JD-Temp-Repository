# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import traceback
import requests

from . import utils
from .trading import LiveTrading
from .user_factor import UserFactor
from .notebook import Notebook
from .data import Data
from .authorize import Authorize

### python 3.x unicode
try:
    unicode
except:
    unicode = str


session = requests.Session()

class Client(Authorize):
    """优矿
    """
    def __init__(self, username='', password='', token='', account_file='', data_target_path='', notebook_target_path=''):

        Authorize.__init__(self, username, password, token, account_file, session)
        
        ### Authorize
        self._authorize()

        if self._isvalid:
            ### Trading
            self.trading = LiveTrading(self.session)

            ### notebook，data，library
            self.data = Data(self.session, data_target_path)
            self.notebook = Notebook(self.session, notebook_target_path)

            ## factor
            self.factor = UserFactor(self.session)
    


def sdk_version(auto_update=False):
    """查询优矿专业版 SDK 版本信息

    参数：
        auto_update: 设置为 True 时可以自动更新，设置为 False 时用户可以根据函数打印的下载链接自行下载安装。
    """
    utils.version_check(auto_update)
