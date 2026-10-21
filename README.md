Here is where I write my small experiments while I learn about machine learning

## Projects

- [building-micrograd](playground/building-micrograd): building micrograd from scratch, an autograd engine with a `Value` class, an `MLP` on top of it, and a spiral classification to test it.
  - [notes.md](playground/building-micrograd/notes.md): my notes on how operations and backward work, and why we sum and reset grads.
  - [micrograd_from_scratch.py](playground/building-micrograd/micrograd_from_scratch.py): the `Value` class and the `MLP`.
  - [spiral_classification.py](playground/building-micrograd/spiral_classification.py): trains the `MLP` on two spirals, saves [classification.png](playground/building-micrograd/classification.png).
  - [visualize_graph.py](playground/building-micrograd/visualize_graph.py): draws the computation graph with graphviz, output in [graph.svg](playground/building-micrograd/graph.svg).
