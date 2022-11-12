# -*- coding:utf-8 -*-
"""
The MIT License (MIT)
Copyright (c) 2022-present D1ffic00lt
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import pickle
import string
import nltk
import re
import os

from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from typing import Tuple, List

__all__ = ("ToxicityClassificator", )


class ToxicityClassificator(object):
    r"""
    A class that determines the degree of toxicity
    of the text using the module "logistic regression"

    classify(sentence: :class:`str`) -> :class:`int`
        a string is given as input, the function processes it
        and returns a tuple from the final result (bool) and
        the probability of this result (float)
    get_probability(sentence: :class:`str`) -> :class:`float`
        the function takes a string as input
        and returns the probability of toxicity
    predict(sentence: :class:`str`) -> :class:`float`
        the function takes a string as input
        and returns a toxic text toxicity status
    download_nlkt() -> None:
        nlkt error-correcting function
    """
    __slots__ = (
        "__russian_model",
        "__english_model",
        "__russian_vectorizer",
        "__english_vectorizer",
        "__russian_snowball",
        "__english_snowball",
        "__weight",
        "__tokens",
        "__toxic_propabality",
        "__language_weight"
    )
    __annotations__ = {
        "language_weight": float,
        "weight": float
    }

    def __init__(self) -> None:
        """
        Returns
        -------
        :exc:`.FileNotFoundError`
            if it happens, then this is a problem,
            there are no model files in the directory,
            I honestly don’t know how to solve it
        """
        with open(f"{os.path.abspath(__file__)[:-21]}models/RussianModel.bf", "rb") as model1, \
                open(f"{os.path.abspath(__file__)[:-21]}models/EnglishModel.bf", "rb") as model2:
            self.__russian_model = pickle.load(model1)
            self.__english_model = pickle.load(model2)
        with open(f"{os.path.abspath(__file__)[:-21]}models/RussianVectorizer.bf", "rb") as model1, \
                open(f"{os.path.abspath(__file__)[:-21]}models/EnglishVectorizer.bf", "rb") as model2:
            self.__russian_vectorizer = pickle.load(model1)
            self.__english_vectorizer = pickle.load(model2)
        self.__russian_snowball: SnowballStemmer = SnowballStemmer(language="russian")
        self.__english_snowball: SnowballStemmer = SnowballStemmer(language="english")
        self.__tokens: List[str] = []
        self.__toxic_propabality: float = -1
        self.__weight: float = 0.5
        self.__language_weight: float = 0.5

    def predict(self, sentence: str) -> Tuple[int, float]:
        """
        Function predicts text toxicity class and probability

        Parameters
        ----------
        sentence :
            string to be processed to further
            predict the degree of toxicity

        Returns
        --------
        Tuple[int, float]
            the function returns the message the probability value

        """
        return self.__get_toxicity(sentence)

    def get_probability(self, sentence: str) -> float:
        """
        Function predicts text toxicity probability

        Parameters
        ---------
        sentence :
            string to be processed to further
            predict the degree of toxicity

        Returns
        --------
        :class:`float`
            the function returns the probability of toxicity

        """
        return self.predict(sentence)[1]

    def classify(self, sentence: str) -> float:
        """
        Function predicts text toxicity value

        Parameters
        ---------
        sentence :
            string to be processed to further
            predict the degree of toxicity

        Returns
        -------
        :class:`float`
            the function returns the message class (1 or 0)
        """
        return self.predict(sentence)[0]

    @staticmethod
    def download_nlkt() -> None:
        """function fixes nlkt error"""
        nltk.download("punkt")
        nltk.download("stopwords")

    def __russian_tokenizer(self, sentence: str) -> str:
        sentence = sentence.strip().lower()
        sentence.replace("\t", " ")
        for i in string.punctuation:
            if i in sentence and i != " ":
                sentence = sentence.replace(i, '')
        try:
            self.__tokens = word_tokenize(sentence, language="russian")
            self.__tokens = [i for i in self.__tokens if i not in stopwords.words("russian")]
            self.__tokens = [self.__russian_snowball.stem(i) for i in self.__tokens]
        except LookupError:
            raise LookupError("use ToxicityClassificator.download_nlkt() before training to fix the error")
        return " ".join(self.__tokens)

    def __english_tokenizer(self, sentence: str) -> str:
        sentence = sentence.strip().lower()
        sentence.replace("\t", " ")
        for i in string.punctuation:
            if i in sentence and i != " ":
                sentence = sentence.replace(i, '')
        try:
            self.__tokens = word_tokenize(sentence, language="english")
            self.__tokens = [i for i in self.__tokens if i not in stopwords.words("english")]
            self.__tokens = [self.__english_snowball.stem(i) for i in self.__tokens]
        except LookupError:
            raise LookupError("use ToxicityClassificator.download_nlkt() before training to fix the error")
        return " ".join(self.__tokens)

    def __get_toxicity(self, sentence: str) -> Tuple[int, float]:
        if len(re.findall(r'[а-я]', sentence)) / len(sentence) >= self.__language_weight:
            self.__toxic_propabality = self.__russian_model.predict_proba(
                self.__russian_vectorizer.transform(
                    [
                        self.__russian_tokenizer(sentence)
                    ]
                )
            )[0, 1]
            return 1 if self.__toxic_propabality >= self.__weight else 0, self.__toxic_propabality
        else:
            self.__toxic_propabality = self.__english_model.predict_proba(
                self.__english_vectorizer.transform(
                    [
                        self.__english_tokenizer(sentence)
                    ]
                )
            )[0, 1]
            return 1 if self.__toxic_propabality >= self.__weight else 0, self.__toxic_propabality

    def __check_models(self) -> bool:
        return True if self.__russian_model is not None and self.__english_model is not None else False

    def __check_vectorizers(self) -> bool:
        return True if self.__russian_vectorizer is not None and self.__english_vectorizer is not None else False

    @property
    def weight(self) -> float:
        return self.__weight

    @weight.setter
    def weight(self, value: float) -> None:
        if 0 >= value >= 1:
            raise ValueError("weight must be between 0 and 1")
        self.__weight = value

    @property
    def language_weight(self) -> float:
        return self.__language_weight

    @language_weight.setter
    def language_weight(self, value: float) -> None:
        if 0 >= value >= 1:
            raise ValueError("language_weight must be between 0 and 1")
        self.__language_weight = value

    def __repr__(self):
        return '%s()' % (self.__class__.__name__, )

    def __new__(cls, *args, **kwargs):
        return object.__new__(ToxicityClassificator)


if __name__ == "__main__":
    print(repr(ToxicityClassificator()))
