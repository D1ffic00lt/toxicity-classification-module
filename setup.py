import nltk

from setuptools import setup

with open("requirements.txt") as file:
    install_requires = [i.strip() for i in file.readlines()]

setup(
    name='toxicityclassifier',
    version='0.1.11',
    description='Module for determining the degree of toxicity of the text',
    author='D1ffic00lt',
    author_email='fdm.filinov@gmail.com',
    packages=['toxicityclassifier'],
    install_requires=install_requires,
)

nltk.download("punkt")
nltk.download('stopwords')
