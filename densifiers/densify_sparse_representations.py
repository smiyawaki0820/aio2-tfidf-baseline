# This script reproduce DSRs proposed by Lin+21.
# Please refer to the following paper for more information of proposed method:
#   - Sheng-Chieh Lin and Jimmy Lin - Densifying Sparse Representations for Passage Retrieval by Representational Slicing (2021)

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
END = "\033[0m"


class DensifierForSparseVector(object):
    def __init__(self):
        pass

    def __arxiv__(self):
        return "https://arxiv.org/abs/2112.04666"

    @staticmethod
    def slice(sparse_vector, n_slices, how):
        v = sparse_vector.shape[1]
        m = n_slices           # num of slices
        n = math.ceil(v / m)   # each dim
        X = np.pad(sparse_vector, ((0, 0), (0, m*n-v)))

        if how == "contiguous":
            return [X[:, i*n:(i+1)*n] for i in range(m)]
        elif how == "stride":
            return [X[:, i::m] for i in range(m)]
        elif how == "random":
            raise NotImplementedError()
        else:
            raise NotImplementedError("You should specify the arg of `how` in [contiguous, stride, random]")

    def densify(self, X, n_slices=768, how="contiguous"):
        """ densify sparse representations """
        Xs = self.slice(X, n_slices, how)
        assert len(Xs) == n_slices
        # pooling
        max_Xs = np.max(np.stack(Xs), axis=-1).transpose()
        idx_Xs = np.argmax(np.stack(Xs), axis=-1).transpose()
        return max_Xs, idx_Xs

    @classmethod
    def gated_inner_product(cls, max_Q, idx_Q, max_Ds, idx_Ds):
        """ return similarities calculated by GIP (eq.7)
          - max_X: 各 slice における max
          - idx_X: 各 slice における argmax
        """
        assert (max_Q.shape == idx_Q.shape) and (max_Ds.shape == idx_Ds.shape)
        assert len(max_Q.shape) == 1 and len(max_Ds.shape) == 2
        assert max_Q.shape[0] == max_Ds.shape[1]
        gate = (idx_Q == idx_Ds)
        return np.matmul(max_Q, np.multiply(gate, max_Ds).T)

    @classmethod
    def retrieval_and_reranking(cls, max_Q, idx_Q, max_Ds, idx_Ds, theta=0.15):
        """ search related documents by means of "retrieval and reranking"
        - Fig.3 で theta に対する latency-memtric を評価
          - theta = [0.05, 0.3] 辺りが良く見えるが、sparse vectorizor に依存する可能性がある
          - ちなみに論文内では SPLADA/uniCOIL の contextualized sparse vectorizer を使用
        """
        # retrieval
        target = (max_Q > theta)
        _max_Q = max_Q[target]
        _idx_Q = idx_Q[target]
        _max_Ds = max_Ds[:, target]
        _idx_Ds = idx_Ds[:, target]
        # reranking
        return cls.gated_inner_product(_max_Q, _idx_Q, _max_Ds, _idx_Ds)



if __name__ == "__main__":
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    # pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    densifier = DensifierForSparseVector()

    sparse_vector = X.toarray()
    max_Ds, idx_Ds = densifier.densify(sparse_vector, n_slices=2)
    
    qid = 0
    max_Q, idx_Q = max_Ds[qid], idx_Ds[qid]
    print(BLUE + f"Query: {corpus[qid]}" + END)

    results = densifier.retrieval_and_reranking(max_Q, idx_Q, max_Ds, idx_Ds)
    for idx, sim in enumerate(results):
        print(YELLOW + f"    {sim:.4f} ... {corpus[idx]}" + END)
