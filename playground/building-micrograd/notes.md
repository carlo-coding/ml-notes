**How to build each operation and why do we have to sum gradients** #anki
Each operation is defined using its dunder method if possible, calculating using the data attribute that holds the number and wrapping the result into the same Value class and saving the children (numbers that were involved in the operation) to keep track and eventually propagate the gradients. For each operation we have to save a function that defines how to calculate the gradients from the current result to the children and save the function in the current node, that's because when we call loss.backward() gradients propagate in reverse so the result of the operation gets the gradient before its components, because the gradient of a child is the gradient of its result times the local derivative. We sum gradients because a parameter is used in several places, so the sum makes sure each contribution is counted, and also because of the chain rule in the multivariable case.

**How does the backward function work** #anki
The idea is simple to translate into code, it is a simple topological sort of the forward graph, the rule is that a node gets added into a list only if the other nodes it depends on (its children) are already on the list, that's why we need to make a recursive call and check if we have visited a node, in code we add to the list after the recursive call has already gone through its children. After we get the list we loop over it in reverse so we start with the very last node assigning 1 to its grad property (the derivative of the loss with respect to itself) and calling \_backward for every node in the reversed list so we propagate the gradients.

**Why do we call backward() on the variable loss** #anki
Because the loss variable is a starting point to track the whole tree, when we expressed the loss function using the Value class its local derivatives got saved, decomposed for each operation through the graph, so calling backward on loss means we are calculating each gradient related to the final loss, which is the value we want to minimize.

**Once we calculate gradients how and why do we optimize parameters** #anki
We optimize them after calculating the grads, we do it to decrease the loss by updating each parameter in the opposite direction of the gradient (scaled by a learning rate), because adding an amount in the direction of the gradient increases the loss.

**Why do we have to reset grads after an optimization step** #anki
Because backward() accumulates gradients, so the new ones would be added to the old ones, and after each optimization step the old gradients are lying because they tell the loss of a network that no longer exists.
