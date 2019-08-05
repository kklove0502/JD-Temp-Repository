# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import requests
import os

from .config import *
from .version import __version__ as v

version = 'sdk_%s'%v

class Notebook(object):

    def __init__(self, session, target_path):
        self.session = session

        if target_path:
            self.target_path = target_path
        else:
            self.target_path = os.getcwd()

        self.notebook = self.list()


    def list(self):
        self.notebook = self._list_notebook()
        return self.notebook


    def _list_notebook(self):
        url = NOTEBOOK_URL
        res = self.session.get(url)
        if not res.ok:
            return 0
        data = res.json()

        if isinstance(data, dict) and data.get('code'):
            print(data)
            return []

        all_notebook = []
        self.directories = []

        for elem in data:
            name = elem.get('name')
            type = elem.get('type')

            if type == 'notebook':
                all_notebook.append(name)

            elif type == 'directory':
                self.directories.append(name)
                sub_elems = elem.get('children', [])
                for sub_elem in sub_elems:
                    elem_name = sub_elem.get('name')
                    type = sub_elem.get('type')
                    if type == 'notebook':
                        all_notebook.append('%s/%s'%(name, elem_name))

        return all_notebook
        

    def backup(self):
        print("该方法已废弃，请使用 client.notebook.download(download_all=True) 下载所有的notebook文件。")


    def download(self, filename='', target_path='', download_all=False):

        if target_path:
            self.target_path = target_path  

        self.notebook = self.list()
        directories = self.directories

        if download_all:
            for i in self.notebook:
                self._download_notebook(i)

        elif type(filename) == list:
            for i in filename:
                # 以ipynb格式请求后端文件
                i = i.replace('.nb', '.ipynb')
                if i in self.notebook:
                    self._download_notebook(i)
                elif i in directories:
                    self._download_dir(i)
                else:
                    print ('文件不存在 {}'.format(i))

        elif type(filename) in (str, unicode):
            # 以ipynb格式请求后端文件
            filename = filename.replace('.nb', '.ipynb')
            if filename in self.notebook:
                self._download_notebook(filename)
            elif filename in directories:
                self._download_dir(filename)
            else:
                print ('文件不存在 {}'.format(filename))
        else:
            pass


    def _download_dir(self, dir):
        dir =dir.replace('/', '')
        dir_elems = [e for e in self.notebook if e.startswith('%s/'%dir)]
        for dir_elem in dir_elems:
            self._download_notebook(dir_elem)


    def _download_notebook(self, filename):
        url = DOWN_NOTEBOOK_URL
        
        notebook_url = url + '/' + requests.utils.quote(filename.encode('utf-8'))

        # 以nb格式保存到本地文件
        filename = filename.replace('.ipynb', '.nb')
        print('下载Notebook：\n开始下载 %s' %filename)

        target_path = self.target_path

        file_path = os.path.join(target_path, filename)

        try:
            path, filename = file_path.rsplit('/', 1)
        except:
            path = ''

        if path and not os.path.exists(path):
            os.mkdir(path)
            print('文件夹 %s 不存在，已自动创建。'%path)

        with open(file_path, 'wb') as f:
            response = self.session.get('%s?%s'%(notebook_url, version), stream=True)

            if not response.ok:
                if response.status_code == 413:
                    print('{}文件超过100兆，暂不支持下载。'.format(filename))
                else:
                    print ('下载文件 {} 过程中出错'.format(filename))
                return 0
            
            for chunk in response.iter_content(1024 * 100):
                f.write(chunk)

        print('完成下载 %s, 路径：%s' %(filename, file_path))


    def delete(self, filename):
        try:
            _delete_notebook(filename)
            print ('删除成功 {}'.format(filename))
        except:
            print ('删除过程出错：', traceback.print_exc())

    def _delete_notebook(self, filename):
        url = NOTEBOOK_DELETE()
        dataurl = url.format(filename)
        self.session.delete('%s?%s'%(dataurl, version))

        print('%s 数据文件删除成功' %(filename))






