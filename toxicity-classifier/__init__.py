
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    raise ImportError("sklearn is not installed. Try \"pip install scikit-learn\"")

try:
    import nltk
except ImportError:
    raise ImportError("nltk is not installed. Try \"pip install nltk\"")
