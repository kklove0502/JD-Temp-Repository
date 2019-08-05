#coding=utf-8
global_cache_map = {}

class Settings(object):
    '''
    DataAPI的设置
    '''
    def __init__(self):
        self._cache_enabled = False
        self.pretty_traceback_enabled = True
    @property
    def cache_enabled(self):
        """DataAPI cache功能是否开启"""
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        if not value:
            global_cache_map.clear()
        self._cache_enabled = value