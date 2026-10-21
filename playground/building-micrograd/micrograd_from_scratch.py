
from pathlib import Path
from visualize_graph import draw_dot
import math
import random
import itertools

class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op
        self.label=label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"^{other}")
        def _backward():
            self.grad += (other * self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        visited = set()
        topo = []
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for v in node._prev:
                    build_topo(v)
                topo.append(node)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        return sum([wi*xi for wi, xi in zip(self.w, x)], self.b).tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return list(itertools.chain(*[neuron.parameters() for neuron in self.neurons]))


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return list(itertools.chain(*[layer.parameters() for layer in self.layers]))
        


if __name__ == "__main__":

    print("\n\n" + "---"*5 + " CHECK OUTPUTS " + "---"*5 + "\n")
    
    print("Neuron check: ", Neuron(2)([2, 3]))
    print("Layer check: ", Layer(2, 3)([3, 2]))
    print("MLP check: ", MLP(3, [4, 3, 1])([3, 2, 5]))


    print("\n\n" + "---"*5 + " TRAINING TEST " + "---"*5 + "\n")

    # DATA
    xs=[
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # MODEL
    n = MLP(3, [2, 1])

    # TRAINING LOOP
    epochs = 10
    lr = 0.2
    for epoch in range(epochs):
        # FORWARD
        ypred = [n(x) for x in xs]

        # BACKWARD
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # UPDATE
        for p in n.parameters():
            p.data += -lr*p.grad

        print("Epoch %i Loss %.4f" % (epoch+1, loss.data))
    
    print(ypred)

    print(Path(__file__).parent)
    draw_dot(loss).render(f"{Path(__file__).parent}/graph")