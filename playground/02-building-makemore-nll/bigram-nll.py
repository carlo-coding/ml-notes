import numpy as np
import string
from pathlib import Path
import random

with open(str(Path(__file__).parent)+"/names.txt") as f:
    data = f.read()

data = data.split("\n")

chars = ["."] + list(string.ascii_lowercase)
stoi = {c: i for i, c in enumerate(chars)}

def train(words):
    N = np.ones((27, 27))

    for w in words:
        cs = ["."] + list(w) + ["."]
        for c1, c2 in zip(cs, cs[1:]):
            N[stoi[c1], stoi[c2]] += 1
    P = N / N.sum(axis=1, keepdims=True)
    return P

def nll(P, words):
    log_likelihood = 0.0
    n = 0

    for w in words:
        cs = ["."] + list(w) + ["."]
        for c1, c2 in zip(cs, cs[1:]):
            prob = P[stoi[c1], stoi[c2]]
            log_likelihood += -np.log(prob)
            n+=1
    return log_likelihood / n

p = train(data)

print(p.shape)
print(p[stoi["a"]])
print(nll(p, data))
print(nll(p, ["andrejq"]))

## Generator, sample 10 names
for i in range(10):
    name = "."
    chosen = ""
    while chosen != ".":
        chosen = random.choices(chars, p[stoi[name[-1]]])[0]
        name += chosen
    print("#%i | Name: %s" %(i, name))

