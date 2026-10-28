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
- [03-building-transformer](playground/03-building-transformer): building a small decoder-only transformer from scratch in PyTorch and training it to generate text character by character.
  - [notes.md](playground/03-building-transformer/notes.md): my notes on attention, causal masking and the scaling before softmax.
  - [main.py](playground/03-building-transformer/main.py): the attention head, multi-head attention, the feed-forward block, the language model and the training and generation loop.
  - [save-data.py](playground/03-building-transformer/save-data.py): downloads the dataset and saves the train and validation splits.
  - [data_train.txt](playground/03-building-transformer/data_train.txt) and [data_val.txt](playground/03-building-transformer/data_val.txt): the training and validation text.
  - [loss.png](playground/03-building-transformer/loss.png): the training loss curve.
  - [generated.txt](playground/03-building-transformer/generated.txt): the training log and a sample of generated text.
  - [notebook.ipynb](playground/03-building-transformer/notebook.ipynb): the notebook I used while building it.
