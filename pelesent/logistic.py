"""
The `pelesent.logistic` module includes LogisticRegression (LR) methods combined with
different text representations. The LR classifier and the TF-IDF representations are both
from sklearn.
"""

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from collections import defaultdict


class TfidfEmbeddingVectorizer(TransformerMixin):
    def __init__(self, model, dim):
        self.model = model
        self.dim = dim
        self.word2weight = None
        self.max_idf = 0

    def fit(self, X, y=None, **fit_params):
        tfidf = TfidfVectorizer(encoding='utf-8')
        tfidf.fit(X)
        self.max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(int, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X, y=None, **fit_params):
        out_m = []

        for text in X:
            parag_M = []
            for token in text.split():
                if token in self.model:
                    if token in self.word2weight:
                        parag_M.append(self.model[token] * self.word2weight[token])
                    else:
                        parag_M.append(self.model[token] * self.max_idf)
            if parag_M:
                out_m.append(np.average(parag_M, axis=0))
            else:
                out_m.append(np.random.rand(1, self.dim)[0])
        return np.array(out_m)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class LogisticRegressionTfidf:

    def __init__(self, transformer='tfidf', embeddings=None):

        if transformer == 'tfidf':
            self.model = make_pipeline(TfidfVectorizer(sublinear_tf=True),
                                       LogisticRegression(random_state=42, n_jobs=4))
        elif transformer == 'tfidf-w2v' and embeddings:
            self.model = make_pipeline(TfidfEmbeddingVectorizer(model=embeddings, dim=embeddings.vector_size),
                                       LogisticRegression(random_state=42, n_jobs=4))
        else:
            print('Transformer not implemented.')

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_prob(self, X):
        return self.model.predict_proba(X)
