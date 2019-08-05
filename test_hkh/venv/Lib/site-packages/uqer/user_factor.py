# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import requests
import os


from uqer.config import Get_Factor, Upload_FactorData, Create_Factor
from uqer.factor import load_df_from_file, check_factor_data


class UserFactor(object):
    """优矿用户因子
    """
    def __init__(self, session):
        super(UserFactor, self).__init__()
        self.session = session

    def _get_uploaded_factor(self):
        """获取上传的因子列表

        """
        r, data = get_factors(self.session)
        if not r:
            print (data)
            return []
        return map(lambda item: {'factor_name': item['factor_name'], 'chinese_name': item['chinese_name'],
                                 'category': item['category'], 'begin_tradedate': str(item['begin_tradedate']),
                                 'end_tradedate': str(item['end_tradedate']), 'pool_name': item['pool_name']},
                   filter(lambda item: True if 'upload_factor' in item and item['upload_factor'] else False, data))

    def get_private_uploaded_factor(self):
        result = self._get_uploaded_factor()
        return filter(lambda item: True if item['pool_name'] == 'private' else False, result)

    def get_factor_name(self, f_id):
        r, data = get_factor(self.session, f_id)
        if not r:
            return None
        return data['factor_name']

    def submit_private_factor(self, factor_name, chinese_name, chinese_description, factor_data_file_path,
                       function_list="winsorize,neutralize,standardize", universe="A"):
        """
        创建一个新的私有因子，并上传因子数据，触发因子报告计算
        :param factor_name: 因子名
        :param chinese_name: 因子中文简称
        :param chinese_description: 因子中文描述
        :param category: 因子分类，用逗号分开多个分类，候选值是：Value,Quality,Momentum,Growth,Sentiment,Return_Risk,Derive,Analyst_FORECAST,Common_Index,Per_share_Indicators,Others
        :param factor_data_file_path: 因子数据文件
        :param function_list: 信号处理函数，用逗号分开多个函数，候选值是：winsorize,neutralize,standardize
        :param universe: 投资域选择，用逗号分开多个投资域，候选值是：A, HS300, ZZ500
        :return: (code, data), code可选范围200,400，403,490,409,500。其中code为200时，data是创建的因子id。code为其他时，data是错误信息提示，
                            其中，400表示有输入错误；403表示没有权限；490表示众筹因子达到一天数量限制；409表示因子名字有冲突；500表示服务器内部错误
        """
        return self._submit_factor(factor_name, chinese_name, chinese_description, '', factor_data_file_path,
                                   function_list, universe, factor_type='private')

    def _submit_factor(self, factor_name, chinese_name, chinese_description, category, factor_data_file_path,
                       function_list="winsorize,neutralize,standardize", universe="A", factor_type='public'):
        """
        创建一个新的因子，并上传因子数据，触发因子报告计算
        :param factor_name: 因子名
        :param chinese_name: 因子中文简称
        :param chinese_description: 因子中文描述
        :param category: 因子分类，用逗号分开多个分类，候选值是：Value,Quality,Momentum,Growth,Sentiment,Return_Risk,Derive,Analyst_FORECAST,Common_Index,Per_share_Indicators,Others;
                        空字符串表示没有分类
        :param factor_data_file_path: 因子数据文件
        :param function_list: 信号处理函数，用逗号分开多个函数，候选值是：winsorize,neutralize,standardize
        :param universe: 投资域选择，用逗号分开多个投资域，候选值是：A, HS300, ZZ500
        :param factor_type: public表示创建到海量因子池，private表示创建到私有因子
        :return: (code, data), code可选范围200,400，403,490,409,500。其中code为200时，data是创建的因子id。code为其他时，data是错误信息提示，
                            其中，400表示有输入错误；403表示没有权限；490表示众筹因子达到一天数量限制；409表示因子名字有冲突；500表示服务器内部错误
        """
        for function in function_list.split(','):
            if function not in ("winsorize", "neutralize", "standardize"):
                msg = '信号处理方法错误：%s' % function
                print (msg)
                return 400, msg
        for item in universe.split(','):
           if item not in ('A', 'HSSLL', 'ZZWLL', 'HS300', 'ZZ500'):
               msg = '投资域选项必须是A，HS300，ZZ500'
               print (msg)
               return 400, msg
        for item in category.split(','):
            if item not in (
            'Value', 'Quality', 'Momentum', 'Growth', 'Sentiment', 'Return_Risk', 'Derive', 'Analyst_FORECAST',
            'Common_Index', 'Per_share_Indicators', 'Others', ''):
                msg = '因子分类错误：%s' % item
                print (msg)
                return 400, msg
        if factor_type not in ('public', 'private'):
            msg = 'factor_type %s 不合理' % factor_type
            print (msg)
            return 400, msg
        print ('正在加载因子csv数据...')
        df, msg = load_df_from_file(factor_data_file_path)
        if df is None:
            print (msg)
            return 400, msg
        print ('加载因子csv数据完成')
        print ('正在校验因子csv数据格式...')
        r, msg = check_factor_data(df)
        if not r:
            print (msg)
            return 400, msg
        print ('校验因子csv数据格式完成')
        print ('正在提交因子...')
        r, f_id = create_factor(self.session, factor_name, category, chinese_name, chinese_description,
                                      function_list.split(','), universe, factor_type)
        if r != 200:
            msg = '创建因子失败：%s' % f_id
            print (msg)
            return r, msg
        print ('提交因子完成\n正在上传因子数据并触发因子报告计算...')
        r, msg = upload_factor_data(self.session, f_id, factor_data_file_path)
        if not r:
            msg = '上传因子数据失败%s，因子id:%s，因子名%s，您后续可以使用更新因子数据方法重新上传因子数据并触发因子报告计算' % (msg, f_id, factor_name)
            print(msg)
            return 500, msg
        import time
        time.sleep(3)
        count = 0
        succ = False
        while True:
            if check_factor_upload_done(self.session, f_id):
                succ = True
                break
            if count >= 50:
                break
            time.sleep(10)
            count += 1
        real_factor_name = self.get_factor_name(f_id)
        if succ:
            print ('上传因子数据完成')
        else:
            print ('上传因子数据遇到问题，可能是服务器正忙，请通过相关API查看上传数据是否到位，或者重新操作')
            msg = '创建因子成功，上传因子数据失败，因子id:%s, 因子名:%s' % (f_id, real_factor_name)
            print (msg)
            return 500, msg
        print ('创建因子成功，因子id:%s, 因子名:%s' %(f_id, real_factor_name))
        return 200, f_id

    def update_private_factor_data(self, factor_name, factor_data_file_path):
        """
        更新私有因子数据，并触发因子报告计算
        :param factor_name: 因子名
        :param factor_data_file_path: 因子数据文件地址
        :return:(code, data), code可选范围200,400，404,500。其中code为200时，data是创建的因子id。code为其他时，data是错误信息提示，
                            其中，400表示有输入错误；404表示因子不存在；500表示服务器内部错误
        """
        return self._update_factor_data(factor_name, factor_data_file_path, factor_type='private')

    def _update_factor_data(self, factor_name, factor_data_file_path, factor_type='public'):
        """
        更新因子数据，并触发因子报告计算
        :param factor_name: 因子名
        :param factor_data_file_path: 因子数据文件地址
        :param factor_type: public表示海量因子，private表示私有因子
        :return: (code, data), code可选范围200,400，404,500。其中code为200时，data是创建的因子id。code为其他时，data是错误信息提示，
                            其中，400表示有输入错误；404表示因子不存在；500表示服务器内部错误
        """
        print ('正在加载因子csv数据...')
        df, msg = load_df_from_file(factor_data_file_path)
        if df is None:
            print (msg)
            return 400,
        print ('加载因子csv数据完成')
        print ('正在校验因子csv数据格式...')
        r, msg = check_factor_data(df)
        if not r:
            print (msg)
            return 400, msg
        print ('校验因子csv数据格式完成\n正在上传因子数据...')
        r, msg = update_uploaded_factor_data(self.session, factor_name, factor_data_file_path, factor_type)
        if not r:
            msg = '上传因子数据失败%s，因子名%s，您后续可以使用更新因子数据方法重新上传因子数据并触发因子报告计算' % (msg, factor_name)
            print (msg)
            return 500, msg
        f_id = msg
        import time
        time.sleep(10)
        count = 0
        succ = False
        while True:
            if check_factor_upload_done(self.session, f_id):
                succ = True
                break
            if count >= 60:
                break
            time.sleep(10)
            count += 1
        if succ:
            print ('上传因子数据完成')
        else:
            msg = '上传因子数据遇到问题，可能是服务器正忙，请通过相关API查看上传数据是否到位，或者重新操作\n因子id:%s, 因子名:%s' % (f_id, factor_name)
            print (msg)
            return 500, msg
        print ('更新因子数据成功，因子id: %s，因子名:%s' % (f_id, factor_name))
        return 200, f_id

def create_factor(session, factor_name, 
	              category, 
	              chinese_name, 
	              chinese_des,
                  function_list=["winsorize", "neutralize", "standardize"], 
                  universe='A',
                  factor_type='public'):
    url = Create_Factor
    input_json = {
                    'pool_name': factor_type,
                    'factor_name': factor_name, 
                    'code': '', 
                    'category': category,
                    'chinese_des': chinese_des, 
                    'upload_factor': True, 
                    'chinese_name': chinese_name,
                    'function_list': function_list, 
                    'universe': universe
                }
    try:
        r = session.post(url, json=input_json)
        status_code, info = r.status_code, r.text
        return status_code, info
    except:
        import traceback
        return 500, '调用上传因子服务失败，%s' % traceback.format_exc()


def get_factors(session):
    url = Get_Factor
    import json
    try:
        r = session.get(url + '?need_max_min_td=1')
        status_code, info = r.status_code, r.text
        if status_code == 200:
            return True, json.loads(info)
        return False, info
    except:
        import traceback
        return False, '获取因子失败，%s' % traceback.format_exc()

def get_factor(session, f_id):
    url = Get_Factor
    import json
    try:
        r = session.get('%s/%s' % (url, f_id))
        status_code, info = r.status_code, r.text
        if status_code == 200:
            return True, json.loads(info)
        return False, info
    except:
        import traceback
        return False, '获取因子失败，%s' % traceback.format_exc()


def upload_factor_data(session, f_id, file_path):
    import zlib
    with open(file_path) as f:
        data = f.read()
        try:
            data = zlib.compress(data)
        except:
            data = zlib.compress(bytes(data, encoding='utf-8'))
        url = Upload_FactorData
        try:
            r = session.post(url + '?f_id=%s' % f_id, data=data)
            status_code, msg = r.status_code, r.text
            if status_code == 200:
                return True, msg
            return False, msg
        except:
            import traceback
            return False, '调用因子数据上传服务失败，%s' % traceback.format_exc()


def check_factor_upload_done(session, f_id):
    try:
        url = Upload_FactorData
        r = session.get(url + '?f_id=%s' % f_id, timeout=30)
        status_code, msg = r.status_code, r.text
        if status_code == 200:
            if msg == 'Done':
                return True
    except:
        pass
    return False


def update_uploaded_factor_data(session, factor_name, file_path, factor_type='public'):
    import zlib
    with open(file_path) as f:
        data = f.read()
        try:
            data = zlib.compress(data)
        except:
            data = zlib.compress(bytes(data, encoding='utf-8'))
        url = Upload_FactorData
        try:
            r = session.put(url + '?f_id=%s&trigger_gen_report=%d&pool_name=%s' % (factor_name, 0, factor_type), data=data)
            status_code, msg = r.status_code, r.text
            if status_code == 200:
                return True, msg
            return False, msg
        except:
            import traceback
            return False, '调用因子数据更新服务失败，%s' % traceback.format_exc()

