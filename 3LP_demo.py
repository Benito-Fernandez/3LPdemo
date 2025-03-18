#-------------------------------------------------------------------------------
# Name:        3LP_demo
# Purpose:     Demonstration of how the weights of a Three-Layer Perceptron
#              affect the output of the network.
#              It is meant to be an instructional/learning environment.
#
# Author:      Benito R. Fernandez
#
# Created:     2021.09.24
# Copyright:   (c) Benito Fernandez, 2005-2025
#                  benito.fernandez@gmail.com
#              (c) The Whisper Company, 2019
#                  benito@TheWhisperCompany.com
#              (c) Machine Essence Corporation, 2023
#                  benito.fernandez@MachineEssence.com
# License:     MIT GPL
#-------------------------------------------------------------------------------

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Activation functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def rbf(x, c=0, s=1):
    return np.exp(-np.power((x - c), 2) / (2 * s**2))

# Mapping of activation function names to functions
ACTIVATION_FUNCTIONS = {
    "linear": linear,
    "sigmoidal": sigmoid,
    "tanh": tanh,
    "RBF": rbf,
}

# Neuron class
class Neuron:
    def __init__(self, activation="tanh"):
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.input_weight = np.random.uniform(-3.1, 2.7)
        self.bias = np.random.uniform(-4.3, 4.0)
        self.output_weight = np.random.uniform(-2.7, 3.2)

    def set_weights(self, input_weight, bias, output_weight):
        self.input_weight = input_weight
        self.bias = bias
        self.output_weight = output_weight

    def output(self, x):
        if self.activation == 'RBF':
            return self.activation(x, self.input_weight, 0.01+abs(self.bias)) * self.output_weight
        else:
            return self.activation(self.input_weight * (x - self.bias)) * self.output_weight

# MLP Class
class MLP:
    def __init__(self, n=3, activation="tanh"):
        self.n = n
        self.neurons = [Neuron(activation) for _ in range(n)]

    def set_weights(self, node_idx, input_weight, bias, output_weight):
        self.neurons[node_idx].set_weights(input_weight, bias, output_weight)

    def output(self, x):
        hidden_outputs = [neuron.output(x) for neuron in self.neurons]
        return sum(hidden_outputs), hidden_outputs

# GUI Class
class MLPApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Three-Layer Perceptron GUI")

        self.mlp = MLP()
        self.selected_node = 1

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        self.frame = tk.Frame(self.master)
        self.frame.pack(pady=10)

        self.n_label = tk.Label(self.frame, text="Number of Hidden Nodes:")
        self.n_label.grid(row=0, column=0)
        self.n_entry = tk.Entry(self.frame)
        self.n_entry.insert(0, "3")
        self.n_entry.grid(row=0, column=1)
        self.n_entry.bind("<Return>", self.update_hidden_nodes)

        self.activation_label = tk.Label(self.frame, text="Activation Function:")
        self.activation_label.grid(row=1, column=0)
        self.activation_combobox = ttk.Combobox(self.frame, values=list(ACTIVATION_FUNCTIONS.keys()))
        self.activation_combobox.current(2) # tanh as default
        self.activation_combobox.grid(row=1, column=1)
        self.activation_combobox.bind("<<ComboboxSelected>>", self.update_activation_function)

        self.node_selector_label = tk.Label(self.frame, text="Select Node:")
        self.node_selector_label.grid(row=2, column=0)
        self.node_selector_combobox = ttk.Combobox(self.frame, values=list(range(1, self.mlp.n+1)))
        self.node_selector_combobox.current(0) # first node as default (in case there is only one)
        self.node_selector_combobox.grid(row=2, column=1)
        self.node_selector_combobox.bind("<<ComboboxSelected>>", self.update_selected_node)

        self.input_weight_slider = tk.Scale(self.master, from_=-5, to=5, label="Input Weight", orient=tk.HORIZONTAL, resolution=0.01, command=self.update_weights)
        self.input_weight_slider.pack(fill=tk.X)
        self.bias_slider = tk.Scale(self.master, from_=-5, to=5, label="Bias", orient=tk.HORIZONTAL, resolution=0.01, command=self.update_weights)
        self.bias_slider.pack(fill=tk.X)
        self.output_weight_slider = tk.Scale(self.master, from_=-5, to=5, label="Output Weight", orient=tk.HORIZONTAL, resolution=0.01, command=self.update_weights)
        self.output_weight_slider.pack(fill=tk.X)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.quit_button = tk.Button(self.master, text="Quit", command=self.master.quit)
        self.quit_button.pack()
        self.update_sliders()

    def update_hidden_nodes(self, event):
        n = int(self.n_entry.get())
        self.mlp = MLP(n=n, activation=self.activation_combobox.get())
        self.node_selector_combobox['values'] = list(range(1, n+1))
        self.node_selector_combobox.current(0)
        self.selected_node = 0
        self.update_plot()

    def update_activation_function(self, event):
        activation = self.activation_combobox.get()
        for neuron in self.mlp.neurons:
            neuron.activation = ACTIVATION_FUNCTIONS[activation]
        self.update_plot()

    def update_selected_node(self, event):
        self.selected_node = int(self.node_selector_combobox.get()) - 1
        self.update_sliders()

    def update_sliders(self):
        neuron = self.mlp.neurons[self.selected_node]
        self.input_weight_slider.set(neuron.input_weight)
        self.bias_slider.set(neuron.bias)
        self.output_weight_slider.set(neuron.output_weight)
        self.update_plot()

    def update_weights(self, event=None):
        input_weight = self.input_weight_slider.get()
        bias = self.bias_slider.get()
        output_weight = self.output_weight_slider.get()
        self.mlp.set_weights(self.selected_node, input_weight, bias, output_weight)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        x = np.linspace(-5, 5, 250)
        y, hidden_outputs = self.mlp.output(x)
        self.ax.plot(x, y, 'r-', label='Output')
        colors = ['b--', 'g--', 'm--', 'y--', 'c--', 'k--']  # Add more colors if needed
        for idx, hidden_output in enumerate(hidden_outputs):
            self.ax.plot(x, hidden_output, colors[idx % len(colors)], label=f'Hidden Node {idx + 1}')
        self.ax.set_ylim(-4, 4)
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MLPApp(root)
    root.mainloop()
