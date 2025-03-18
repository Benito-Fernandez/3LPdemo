# 3LPdemo

This is a simple Python code that shows the contribution of each of the weights of any of the hidden nodes of a 3LP (Three-Layer Perceptron).
The 3LP is a 1xnx1 MLP (Multi-Layer Perceptron), with 1 input, n hidden nodes and 1 output.
It is meant to be an instructional/learning environment.

The output layer has a linear activation function and a bias that is the mean of the output in the data set.
The hidden layer activation function can be one of:
def linear(x):
    return x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def rbf(x, c=0, s=1):
    return np.exp(-np.power((x - c), 2) / (2 * s**2))

The GUI shows 
- a field to specify how many hidden nodes are (default is 3),
- a pull-down menu to select which activation function the hidden layer has (default is tanh),
- a pull-down menu to select which selects which hidden node you want to "play" with (default is 1).

 Then, three scroll bars are shown for 
 - the selected hidden node input (input-to-hidden) Weight,
 - the selected hidden node Bias, and
 - the selected hidden node output (hidden-to-output) Weight.
(the defaults is random between min and max values).

The plot shows each hidden node contribution to the output and the network output.

Notes:
- The network output is given by:
    - $y=b_0 + \sum_{i=1}^{i=n} V_i * z_i$,
      where, $z_i = \sigma(xi_i)$, is the hidden node's output value;
      where, $\xi_i = \sigma(b_i + W_i*x)$, in the hidden node's activation function.
- The input weight, $W_i$, is the coeficient that multiplies the input, x, before the activation function, $\sigma()$, takes place.
- The bias, $b_i$, shifts the "activation region" for the node. Combined with $W_i$
- The output weight, $V_i$, is the coeficient that multiplies the hidden node output, z, to contribute to the network's output. It provides the scaling factor for the hidden node's output, $z_i$.
