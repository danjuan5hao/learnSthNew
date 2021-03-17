import numpy as np
from cvxopt import matrix, solvers
from collections import Counter

class Embedding:
    def __init__(self, embedding_file_path):
        self._load(embedding_file_path)

    def _w_ix_v_gen(self, file):
        for ix, line in enumerate(file):
            word_and_vector = line.split()
            word = word_and_vector[0]
            vector = np.array(word_and_vector[1:], dtype=float) 
            yield word, ix, np.array(vector, dtype=float)
    

    def _load(self, path):
        with open(path, "r", encoding="utf8") as f:
            words, ixs, vectors = zip(*self._w_ix_v_gen(f))
            self.v2i = dict(zip(words, ixs))
            self.i2v = dict(zip(ixs, words))
            self.emb = np.vstack(vectors)


class Doc:
    def __init__(self, doc, embed):
        self.text = doc
        self.embed = embed
        self.tokened = self._preprocess(doc).split(" ")
        self.id_seq = [embed.v2i(token) for token in self.tokened]

        self.counter = {k: v/ len(self.tokened) for k, v in Counter(self.id_seq).items()}
        self.nBow = self._get_nbow(self.counter)
        self.id_nBow, self.d_nbow = zip(*self.nBow)

    def __len__(self):
        return len(self.counter)

    def _get_nbow(self, counter):
        return sorted([(self.embed.v2i(word), d) for word, d in counter.items()], key=lambda x: x[0])
        

    def _preprocess(self, text):
        return text

class WMD:
    def __init__(self, embed, distance="euclid"):
        self.embed = embed
        pass 
    
    def _wd(self, vec1, vec2, how="euclid"):
        return np.linalg.norm(vec1 - vec2)

    def _get_minimize_modulus(self, doc1, doc2):
        modulus = []
        for i in doc1.id_nBow:
            i_emb = self.embed[i]
            # b = []
            for j in doc2.id_nBow:
                j_emb = self.embed[j]
                modulus.append(self._wd(i_emb, j_emb))
            # a.append(b)
        return modulus

    def _get_a_eq_aux(self, doc1, doc2):
        return 
        pass

    def _get_a_eq(self, doc1, doc2):
        # 一共有 len(doc1) + len(doc2)个等式约束
        # 一共有 len(doc1) * len(doc2)个参数
        n_param = len(doc1) * len(doc2) 
        n_subject = len(doc1) + len(doc2)
        a_eq = np.zeros([n_param, n_subject], dtype=float)

        # di的约束
        for i in range(len(doc1)):
            start_idx = i*len(doc2)
            end_idx = end_idx + len(doc2)
            a_eq[i, start_idx: end_idx] = 1.0

        for j in range(len(doc2)):
            strat_idx = lenj*len(doc1)
            end_idx
        return 

    def _get_b_eq(self, doc1, doc2):
        # 一共有 len(doc1) + len(doc2)个等式约束
        # 一共有 len(doc1) * len(doc2)个参数
        return np.vstack(doc1.d_nbow.T, doc2.d_nbow.T) 

    def wmd(self, sent1, sent2):
        doc1 = Doc(sent1, self.embed)
        doc2 = Doc(sent2, self.embed)

        #
        A_eq = [[1]]
        b_eq = []

        bounds = [(0, None)] * (len(doc1) *len(doc2))

        c = np.array(self._get_minimize_modulus(doc1, doc2), dtype=float)
        pass 

def wmd(sent1, sent2):
    return 

if __name__ == "__main__":
    file = r"D:\data\学习资料\公开课\贪心学院高级nlp\课件\glove.6B\glove.6B.100d.txt"

    glove = Embedding(file)
    # print(glove.i2v.get(0))
    # print(glove.v2i.get('the'))
    # print(glove.emb[0])
    

