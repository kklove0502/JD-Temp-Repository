# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import dateutil.parser as dt_parser
from .config import *


def convert_date(date, format='%Y-%m-%d'):
    try:
        if isinstance(date, (str, unicode)):
            date = dt_parser.parse(date)
    except Exception as e:
        raise Exception('date:{}格式不能识别。'%date)

    return date.strftime(format)
    



