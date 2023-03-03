# -*- coding:utf-8 -*-
"""
Python Encoder
~~~~~~~~~~~~~~~~~~~
A basic wrapper for the Discord API.
:copyright: (c) 2023-present Dmitry Filinov (D1ffic00lt)
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

__version__ = "0.1.11pre"
__title__ = 'ToxicityClassificator'
__author__ = "Dmitry Filinov (D1ffic00lt)"
__copyright__ = "Copyright 2022-2023 {}".format(__author__)
__all__ = (
    "ToxicityClassificatorV1",
)
