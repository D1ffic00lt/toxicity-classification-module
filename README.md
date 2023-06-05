# ToxicityClassificator
Module for predicting toxicity messages in Russian and English
## Usage example
```python
from toxicityclassifier import *

classifier = ToxicityClassificatorV1()

print(classifier.predict(text))          # (0 or 1, probability)
print(classifier.get_probability(text))  # probability
print(classifier.classify(text))         # 0 or 1
```

## Weights
Weight for classification (if probability >= weight => 1 else 0)
```python
classifier.weight = 0.5
```
\
Weight for language detection (English or Russian)

if the percentage of the Russian language >= language_weight, then the Russian model is used, otherwise the English one
```python
classifier.language_weight = 0.5
```
