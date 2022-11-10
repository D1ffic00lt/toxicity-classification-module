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

from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from typing import Tuple, List


class ToxicityClassificator(object):
    r"""
    A class that determines the degree of toxicity
    of the text using the module "logistic regression"

    .. versionadded:: 1.0.0

    Methods
    --------
    classify(sentence: :class:`str`) -> :class:`int`
        a string is given as input, the function processes it
        and returns a tuple from the final result (bool) and
        the probability of this result (float)
    get_probability(sentence: :class:`str`) -> float:
        the function takes a string as input
        and returns the probability of toxicity
    predict(sentence: :class:`str`) -> :class:`float`
        the function takes a string as input
        and returns a toxic text toxicity status
    download_nlkt() -> None:
        nlkt error-correcting function
    """
    def __init__(self) -> None:
        """
        Returns
        --------
        :exc:`.FileNotFoundError`
            if it happens, then this is a problem,
            there are no model files in the directory,
            I honestly don’t know how to solve it
        """
        self._russian_snowball: SnowballStemmer = SnowballStemmer(language="russian")
        self._english_snowball: SnowballStemmer = SnowballStemmer(language="english")
        self._sentence: str = ""
        self._tokens: List[str] = []
        self._russian_model: pickle.bytes_types = None
        self._english_model: pickle.bytes_types = None
        self._russian_vectorizer: pickle.bytes_types = None
        self._english_vectorizer: pickle.bytes_types = None
        self._toxic_propabality: float = 6.
        self._weight: float = 0.5
        self._language_weight: float = 0.5

    def predict(self, sentence: str) -> Tuple[int, float]:
        """
        Function predicts text toxicity class and probability

        Parameters
        --------
        :param sentence: :class:`str`
            string to be processed to further
            predict the degree of toxicity

        Returns
        --------
        Tuple[int, float]
            the function returns the message the probability value

        """
        if not self.__check_models():
            self.__get_models()
        if not self.__check_vectorizers():
            self.__get_vectorizers()
        return self.__get_toxicity(sentence)

    def get_probability(self, sentence: str) -> float:
        """
        Function predicts text toxicity probability

        Parameters
        --------
        :param sentence: :class:`str`
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
        --------
        :param sentence: :class:`str`
            string to be processed to further
            predict the degree of toxicity

        Returns
        --------
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
        self._sentence = sentence.strip().lower()
        self._sentence.replace("\t", " ")
        for i in string.punctuation:
            if i in self._sentence and i != " ":
                self._sentence = self._sentence.replace(i, '')
        try:
            self._tokens = word_tokenize(self._sentence, language="russian")
            self._tokens = [i for i in self._tokens if i not in stopwords.words("russian")]
            self._tokens = [self._russian_snowball.stem(i) for i in self._tokens]
        except LookupError:
            raise LookupError("use ToxicityClassificator.download_nlkt() before training to fix the error")
        return " ".join(self._tokens)

    def __english_tokenizer(self, sentence: str) -> str:
        self._sentence = sentence.strip().lower()
        self._sentence.replace("\t", " ")
        for i in string.punctuation:
            if i in self._sentence and i != " ":
                self._sentence = self._sentence.replace(i, '')
        try:
            self._tokens = word_tokenize(self._sentence, language="english")
            self._tokens = [i for i in self._tokens if i not in stopwords.words("english")]
            self._tokens = [self._english_snowball.stem(i) for i in self._tokens]
        except LookupError:
            raise LookupError("use ToxicityClassificator.download_nlkt() before training to fix the error")
        return " ".join(self._tokens)

    def __get_toxicity(self, sentence: str) -> Tuple[int, float]:
        if len(re.findall(r'[а-я]', sentence)) / len(sentence) >= self._language_weight:
            self._toxic_propabality = self._russian_model.predict_proba(
                self._russian_vectorizer.transform(
                    [
                        self.__russian_tokenizer(sentence)
                    ]
                )
            )[0, 1]
            return 1 if self._toxic_propabality >= self._weight else 0, self._toxic_propabality
        else:
            self._toxic_propabality = self._english_model.predict_proba(
                self._english_vectorizer.transform(
                    [
                        self.__english_tokenizer(sentence)
                    ]
                )
            )[0, 1]
            return 1 if self._toxic_propabality >= self._weight else 0, self._toxic_propabality

    def __get_models(self) -> None:
        with open("models/RussianModel.bf", "rb") as model1, \
                open("models/EnglishModel.bf", "rb") as model2:
            self._russian_model = pickle.load(model1)
            self._english_model = pickle.load(model2)

    def __get_vectorizers(self) -> None:
        with open("models/RussianVectorizer.bf", "rb") as model1, \
                open("models/EnglishVectorizer.bf", "rb") as model2:
            self._russian_vectorizer = pickle.load(model1)
            self._english_vectorizer = pickle.load(model2)

    def __check_models(self) -> bool:
        return True if self._russian_model is not None and self._english_model is not None else False

    def __check_vectorizers(self) -> bool:
        return True if self._russian_vectorizer is not None and self._english_vectorizer is not None else False

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        if 0 >= value >= 1:
            raise ValueError("weight must be between 0 and 1")
        self._weight = value

    @property
    def language_weight(self) -> float:
        return self._weight

    @language_weight.setter
    def language_weight(self, value: float) -> None:
        if 0 >= value >= 1:
            raise ValueError("language_weight must be between 0 and 1")
        self._language_weight = value


if __name__ == "__main__":
    print(ToxicityClassificator().predict(input("Enter text: ")))
    input()
