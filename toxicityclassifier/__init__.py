# -*- coding:utf-8 -*-
"""
Python Encoder
~~~~~~~~~~~~~~~~~~~
A basic wrapper for the Discord API.
:copyright: (c) 2022-present D1ffic00lt
:license: MIT, see LICENSE for more details.
"""
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    raise ImportError("sklearn is not installed. Try \"pip install scikit-learn\"")

try:
    import nltk
except ImportError:
    raise ImportError("nltk is not installed. Try \"pip install nltk\"")

from toxicityclassifier.toxicityclassifier import *

__version__ = "0.1.5"
__title__ = 'ToxicityClassificator'
__author__ = "D1ffic00lt"
__copyright__ = "Copyright 2013-2022 {}".format(__author__)
__all__ = ("ToxicityClassificator", )
