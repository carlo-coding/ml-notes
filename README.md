Here is where I write my small experiments while I learn about machine learning

## Projects

- [01-building-micrograd](playground/01-building-micrograd): building micrograd from scratch, an autograd engine with a `Value` class, an `MLP` on top of it, and a spiral classification to test it.
  - [notes.md](playground/01-building-micrograd/notes.md): my notes on how operations and backward work, and why we sum and reset grads.
  - [micrograd_from_scratch.py](playground/01-building-micrograd/micrograd_from_scratch.py): the `Value` class and the `MLP`.
  - [spiral_classification.py](playground/01-building-micrograd/spiral_classification.py): trains the `MLP` on two spirals, saves [classification.png](playground/01-building-micrograd/classification.png).
  - [visualize_graph.py](playground/01-building-micrograd/visualize_graph.py): draws the computation graph with graphviz, output in [graph.svg](playground/01-building-micrograd/graph.svg).
- [02-building-makemore-nll](playground/02-building-makemore-nll): building a character-level bigram model from scratch on a list of names, and measuring how well it fits with the negative log likelihood.
  - [notes.md](playground/02-building-makemore-nll/notes.md): my notes on what a bigram is and what the negative log likelihood measures.
  - [bigram-nll.py](playground/02-building-makemore-nll/bigram-nll.py): trains the bigram count matrix on `names.txt`, computes the NLL, and samples new names.
  - [names.txt](playground/02-building-makemore-nll/names.txt): the list of names the model is trained on.
