from pathlib import Path
import pickle
from time import time
from datetime import timedelta
# Spanish data taken from the transcripts of this playlist:
#   https://www.youtube.com/playlist?list=PLL1LurcyuUeKSDzfTNW6iegT7NJwOYuDx

start = time()

class BPETokenizer:
    def __init__(self):
        self.vocab = set()
        self.merges = {}


    def _get_stats(self, ids):
        counts = {}
        for id0, id1 in zip(ids, ids[1:]):
            counts[(id0, id1)] = counts.get((id0, id1), 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and pair[0] == ids[i] and pair[1] == ids[i+1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, num_merges=100):
        if type(text) != str:
            raise TypeError("Tokenizer can only train on text data")
            
        ids = list(text.encode("utf-8"))
        tok_idx = 255

        while tok_idx < (255 + num_merges):
            stats = self._get_stats(ids)
            new_merge = max(stats, key=stats.get)
            tok_idx += 1
            self.merges[new_merge] = tok_idx
            ids = self._merge(ids, new_merge, tok_idx)

        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for (id0, id1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[id0] + self.vocab[id1]

        return self

    def encode(self, text):
        if type(text) != str:
            raise TypeError("Tokenizer can only train on text data")
        ids = list(text.encode("utf-8"))

        while len(ids) > 1:
            pairs = self._get_stats(ids)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])

        return ids
    
    def decode(self, ids):
        data = []
        for id in ids:
            data.append(self.vocab[id])
        return b"".join(data).decode("utf-8", errors="replace")


with open(str(Path(__file__).parent) + "/data-youtube-es.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=1000)

print("Completed tokenization of %i characters in %s" % (len(corpus), str(timedelta(seconds=time() - start))))

with open("tokenizer.pkl", "wb") as f:
    pickle.dump({
        "vocab": tokenizer.vocab,
        "merges": tokenizer.merges
    }, f)


text = """
En lo profundo del océano existe un mundo fascinante que todavía guarda
innumerables secretos. Allí, criaturas extraordinarias desarrollan formas
sorprendentes de sobrevivir en completa oscuridad, mientras exploradores
y científicos intentan comprender los misterios de uno de los lugares
menos conocidos de nuestro planeta.
"""

tokens = tokenizer.encode(text)

print("Tokens:", tokens)
print("Decoded text:", tokenizer.decode(tokens))

# Next Steps
# [ ] Add more training documents
# [ ] Send corpus as an array of documents
# [ ] Add token visualization
# [ ] Train a tiny transformer on the data