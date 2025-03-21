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

![Screenshot 2025-03-18 121357](https://github.com/user-attachments/assets/91ba7edc-7476-4ab9-abbb-a0c80898c46d)

 Then, three sliders are shown for 
 - the selected hidden node input (input-to-hidden) Weight,
 - the selected hidden node Bias, and
 - the selected hidden node output (hidden-to-output) Weight.
(the defaults is random between min and max values).

The plot shows each hidden node contribution to the output and the network output.

Notes:
- The network output, $y$, is given by:
    - $y=b_0 + \sum_{i=1}^{i=n} V_i * z_i$,
    - $z_i = \sigma(\xi_i)$, is the hidden node's output value, it's activation;
    - $\xi_i = b_i + W_i*x$, in the hidden node's input, before activation;
    - and $\sigma()$ is the node's activation function.
- The input weight, $W_i$, is the coeficient that multiplies the input, $x$, before the activation function, $\sigma()$, takes place.
- The bias, $b_i$, shifts the "activation region" for the node, combined with $W_i$.  The "center" of the activation is when $\xi_i=0$, which is at $c_i=-b_i/W_i$.
- The output weight, $V_i$, is the coeficient that multiplies the hidden node output, $z$, to contribute to the network's output. It provides the scaling factor for the hidden node's output, $z_i$.
- The network bias, $b_0$ moves the network's output up and down.  In this demo, $b_0=0$ for simplicity.

The user, by moving the sliders, can see the influence of each.

Feel free to modify the code or add other activation functions.
