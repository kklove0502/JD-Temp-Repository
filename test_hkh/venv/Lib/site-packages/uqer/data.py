# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import json

import requests
import os

from .config import *
from .version import __version__ as v

version = 'version=sdk_%s'%v


def format_header_param(name, value):
    value = '%s=%s' % (name, value)
    return value


class Data(object):

    def __init__(self, session, data_path):
        self.session = session
        
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = os.getcwd()

        self.data = self.list()


    def backup(self):
        print("该方法已废弃，请使用 client.notebook.download(download_all=True) 下载所有的data文件。")


    def upload(self, filepath, target_path=''):

        self.data = self.list()

        if target_path.endswith('/'):
            target_path = target_path[:-1]

        if target_path.find('/')>=0:
            return '目标路径只支持一级文件夹，请检查/字符。'

        if target_path and target_path not in self.directories:
            self._mk_dir(target_path)

        filepath = os.path.join(self.data_path, filepath)

        try:
            file_info = os.stat(filepath)
            if file_info.st_size>=100*1024*1024:
                print('{}文件超过100兆，暂不支持上传。'.format(filepath))
                return

            f = open(filepath, 'rb')
        except:
            print ("无法打开文件 {}，可能是权限不足或者文件不存在".format(filepath))
            return

        self._upload_data(filepath, target_path)


    def _upload_data(self, filepath, target_path):

        old_format_param = requests.packages.urllib3.fields.format_header_param
        requests.packages.urllib3.fields.format_header_param = format_header_param

        if target_path:
            url = '%s/%s'%(MERCURY_URL, target_path)
        else:
            url = MERCURY_URL

        try:
            files = {'datafile': open(filepath, 'rb')}
            values = {}
            r = self.session.post('%s?%s'%(url, version), files=files, data=values).json()
            if 'name' in r:
                print ('数据文件 {} 上传成功'.format(filepath))
            else:
                print ('数据文件 {} 上传失败'.format(filepath))
                print (r)
        except:
            traceback.print_exc()
        finally:
            requests.packages.urllib3.fields.format_header_param = old_format_param
        

    def list_data(self):
        url = MERCURY_URL
        res = self.session.get('%s?recursion'%(url))
        if not res.ok:
            print ('请求超时，请重试，重试未成功请直接联系客服.')
            return 0

        data = res.json()
        if isinstance(data, dict) and data.get('code'):
            print(data)
            return []

        all_data = []
        self.directories = []

        for elem in data:
            name = elem.get('name')
            type = elem.get('type')

            if type == 'databook':
                all_data.append(name)

            elif type == 'directory':
                self.directories.append(name)

                sub_elems = elem.get('children', [])
                for sub_elem in sub_elems:
                    elem_name = sub_elem.get('name')
                    type = sub_elem.get('type')
                    if type == 'databook':
                        all_data.append('%s/%s'%(name, elem_name))

        return all_data


    def list(self):
        self.data = self.list_data()
        return self.data


    def download(self, filename='', target_path='', download_all=False):

        self.data = self.list()
        directories = self.directories

        if download_all:
            for i in self.data:
                self._download_file(i, target_path)

        elif type(filename) == list:
            for i in filename:
                if i in self.data:
                    self._download_file(i, target_path)
                elif i in directories:
                    self._download_dir(dir=i, target_path=target_path)
                else:
                    print ('文件不存在 {}'.format(i))

        elif isinstance(filename, (str, unicode)):
            if filename in self.data:
                self._download_file(filename, target_path)
            elif filename in directories:
                    self._download_dir(dir=filename, target_path=target_path)
            else:
                print ('文件不存在 {}'.format(filename))

        else:
            pass


    def _download_dir(self, dir, target_path):
        dir =dir.replace('/', '')
        dir_elems = [e for e in self.data if e.startswith('%s/'%dir)]
        for dir_elem in dir_elems:
            self._download_file(dir_elem, target_path)


    def _download_file(self, filename, target_path):
        url = DATA_URL

        dataurl = url + '/' + filename

        print('下载data：\n开始下载 ', filename)

        if target_path:
            data_path = target_path
        else:
            data_path = self.data_path

        file_path = os.path.join(data_path, filename)

        try:
            path, filename = file_path.rsplit('/', 1)
        except:
            path = ''

        if path:
            if not os.path.exists(path):
                os.mkdir(path)
                print('文件夹 %s 不存在，已自动创建。'%path)
            elif not os.path.isdir(path):
                print('%s非文件夹，请检查文件或修改路径。'%path)
                return

        with open(file_path, 'wb') as f:
            size_info = self.session.get('%s?%s&size=1'%(dataurl, version)).json()
            if size_info['size'] > size_info['today_volume']:
                print ('本功能限制每天下载的数据文件总量为100兆，' \
                      '您今天已经下载%d兆，该文件大小超出限制无法下载。' % (100 - size_info['today_volume']/1024/1024))
                return 0
            response = self.session.get('%s?%s'%(dataurl, version), stream=True)

            if not response.ok:
                if response.status_code == 413:
                    print('{}文件超过100兆，暂不支持下载。'.format(filename))
                else:
                    print ('下载文件 {} 过程中出错'.format(filename))
                return 0
            
            for chunk in response.iter_content(1024 * 100):
                print("...", end="")
                f.write(chunk)

        print('')

        print('完成下载 %s, 路径 %s' %(filename, file_path))


    def delete(self, filename):

        self.data = self.list()
        directories = self.directories

        if isinstance(filename, list):
            for f in filename:
                self._delete_file(f)
        elif isinstance(filename, (str, unicode)):
            if filename in directories:
                self._delete_dir(filename)
            else:
                self._delete_file(filename)


    def _delete_dir(self, dir):
        dir =dir.replace('/', '')
        dir_elems = [e for e in self.data if e.startswith('%s/'%dir)]
        for dir_elem in dir_elems:
            self._delete_file(dir_elem)


    def _delete_file(self, filename):
        url = MERCURY_URL
        dataurl = url + '/' + filename

        try:
            response = self.session.delete('%s?%s'%(dataurl, version))
            status_code = response.status_code
            if status_code == 204:
                print ('{}数据文件删除成功'.format(filename))
            elif status_code == 404:
                print('{}数据文件不存在'.format(filename))
            else:
                print (response.text)
        except:
            print ('删除过程出错：')
            traceback.print_exc()


    def _mk_dir(self, dir_name):
        url = MERCURY_URL
        dataurl = '%s?dir=%s' %(url, dir_name)

        try:
            self.session.post('%s?%s'%(dataurl, version))
            print('文件夹 %s不存在，已自动创建。'%dir_name)
        except:
            print ('创建文件夹出错：')
            traceback.print_exc()
        


















