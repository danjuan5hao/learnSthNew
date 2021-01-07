import numpy as np

class Vocab:
    def __init__(self, file):
        self.i2s, self.s2i, self.embedding = self._build_vocab(file)

    def _build_vocab(self, file):
        with open(file, "r", encoding="utf-8") as f:
            i2s = {}
            s2i = {}
            lines = f.readlines()
            self.num_emb = int(lines[0].split(" ")[0])
            self.emb_dim = int(lines[0].split(" ")[1])

            embedding = np.zeros((self.num_emb, self.emb_dim), dtype=float)

            for idx, line in enumerate(lines[1:]):
                line_split = line.split()
                word = line_split[0]
                vector = np.array(line_split[1:], dtype=float)

                i2s.update({idx: word})
                s2i.update({word: idx})
                embedding[idx] = vector

            return i2s, s2i, embedding
        

if __name__ == "__main__":
    fastttext_file = "fasttext/cc.zh.300.vec"
    fasttext = Vocab(fastttext_file)

    with open("vocab.text", "w", encoding="utf-8", newline="") as f:
        for i in fasttext.s2i:
            f.write(f"{i}\n")
    # print(fasttext.i2s.get(2))
    # print(fasttext.s2i.get("爱情"))