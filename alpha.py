#!/usr/bin/python3

import random
import math
import numpy as np


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


########################################################################################################################
# Input & Data Normalization


class Input(object):
    def __init__(self, input):
        self.input = input

    def normalize(self):
        pass
        # return self.input / self.maximum

    def denormalize(self):
        pass
        # return self.input * self.maximum

    @property
    def input(self):
        return self.input

    @input.setter
    def input(self, input):
        self.input = input


class CategoricalInput(Input):
    def __init__(self, input):
        super().__init__(input)


class NumericalInput(Input):
    def __init__(self, input):
        super().__init__(input)

    @property
    def maximum(self):
        raise NotImplementedError("Input class requires a maximum property to be defined!")


class PriceInput(NumericalInput):
    def __init__(self, input):
        super().__init__(input)

    @property
    def maximum(self):
        return 100


class CategoryInput(CategoricalInput):
    pass


class DiscountInput(NumericalInput):
    def __init__(self, input):
        super().__init__(input)


########################################################################################################################
# Neural Network

class Connection(object):
    def __init__(self):
        self.weight = random.random()
        self.delta  = 0.0


class Neuron(object):
    def __init__(self, nid, lid=0):
        self.nid         = nid
        self.lid         = lid
        self.output      = 0.0
        self.connections = []
        self.gradient    = 0.0
        self.activate    = None
        self.derivate    = None
    
    def weights(self, weights=None):
        if weights is None:
            output = [0.0] * len(self.connections)
            for c in range(len(self.connections)):
                output[c] = self.connections[c].weight
            return output
        else:
            assert len(weights) == len(self.connections)
            for c in range(len(self.connections)):
                self.connections[c].weight = weights[c]


class Network(object):
    eta   = 0.2
    alpha = 0.5
    scale = 0.05

    def __init__(self):
        self.layers    = []
        self.error     = 0.0
        self.average   = 0.0
        self.smoothing = 100.0
        self.outputs   = []

    def setup(self):
        if Network.scale is not None:
            for l in range(1, len(self.layers)):  # 1: skip input
                for n in range(len(self.layers[l])):  # w/o bias
                    for c in range(len(self.layers[l][n].connections)):
                        self.layers[l][n].connections[c].weight *= Network.scale
        Network.scale = None

    def train(self, inputs, targets, batch_size=1):
        self.setup()
        assert batch_size > 0
        if batch_size > 1:  # batch_size = 2, 3, 4, ...  # inputs=((1,2,3),(2,3,4)) targets=((1,2),(2,3))
            pass
            print('TODO: implement logic for training in batches')
        else:                # batch_size = 1              # inputs=(1,2,3)           targets=(1,2)
            assert len(inputs)  == len(self.layers[0])-1   # input values == input neurons(-1 bias)
            assert len(targets) == len(self.layers[-1])-1  # target values == output neurons(-1 bias)
            # set input values to input neurons
            for i in range(len(inputs)):
                self.layers[0][i].output = inputs[i]
            # feed forward to hidden
            for l in range(1, len(self.layers)):         # 1: skip input
                for n in range(len(self.layers[l])-1):   # w/o bias
                    Network.forward(self.layers[l-1], self.layers[l][n])
            # outputs after feed forward
            self.outputs.clear()
            for n in range(len(self.layers[-1]) - 1):  # w/o bias(-1)
                self.outputs.append(self.layers[-1][n].output)
            # calculate overall error(RMS)
            self.error = 0.0
            for n in range(len(self.layers[-1])-1):
                delta = targets[n] - self.layers[-1][n].output
                self.error += delta*delta
            self.error  /= len(self.layers[-1])-1
            self.error   = math.sqrt(self.error)      # RMS
            self.average = (self.average * self.smoothing + self.error) / (self.smoothing + 1.0)
            # back propagate from output to 1st hidden
            # calculate output layer gradients
            for n in range(len(self.layers[-1])-1):   # w/o bias(-1)
                Network.gradient(self.layers[-1][n], targets[n])  # output gradients
            # calculate hidden layer gradients
            for l in range(len(self.layers)-2,0,-1):  # from last hidden layer -> the first hidden layer [hn...h0]
                for n in range(len(self.layers[l])):  # loop each neuron...calc gradinet using next layer neurons
                    Network.gradient(self.layers[l][n], self.layers[l+1])
            # update hidden layer outputs
            for l in range(len(self.layers)-1,0,-1):    # from output layer -> first hidden layer [o...h0]
                for n in range(len(self.layers[l])-1):  # w/o bias(-1)
                    Network.update(self.layers[l-1], self.layers[l][n])  # should it be Layer.udpate(const neuron) ?
            # return output layer outputs
            return self.outputs

    def predict(self, inputs, batch_size=1):
        assert batch_size > 0
        if batch_size > 1:
            pass
        else:
            assert len(inputs) == len(self.layers[0]) - 1  # input values == input neurons(-1 bias)
            # set input values to input neurons
            for n in range(len(inputs)):                   # 0 # set input values
                self.layers[0][n].output = inputs[n]
            # feed forward to hidden
            for l in range(1, len(self.layers)):           # [1..output] # input layer already done
                for n in range(len(self.layers[l]) - 1):   # w/o bias
                    Network.forward(self.layers[l-1], self.layers[l][n])
            # read outputs from the last layer
            self.outputs.clear()
            for n in range(len(self.layers[-1]) - 1):  # w/o bias(-1)
                self.outputs.append(self.layers[-1][n].output)
            return self.outputs

    @staticmethod
    def forward(layer, neuron):  # forward input from prev layer to neuron
        assert type(neuron) is Neuron  # move these 2 asserts
        assert type(layer) is list
        total = 0.0
        for n in range(len(layer)):  # including bias
            total += layer[n].output * layer[n].connections[neuron.nid].weight
        neuron.output = neuron.activate(total)

    @staticmethod
    def gradient(neuron, target):  # target or next layer
        if type(target) is list:
            total = 0.0
            for n in range(len(target)-1):  # w/o bias(-1)
                total += neuron.connections[n].weight * target[n].gradient
            neuron.gradient = total * neuron.derivate(neuron.output)    # output neuron gradient
        else:
            delta = target - neuron.output
            neuron.gradient = delta * neuron.derivate(neuron.output)    # hidden neuron gradient

    @staticmethod
    def update(layer, neuron):  # update layer using a neuron(from next layer)
        for n in range(len(layer)): # prev layer
            olddelta = layer[n].connections[neuron.nid].delta
            newdelta = Network.eta * layer[n].output * neuron.gradient + Network.alpha * olddelta
            layer[n].connections[neuron.nid].delta   = newdelta
            layer[n].connections[neuron.nid].weight += newdelta

    @staticmethod
    def cost(targets, outputs):
        assert len(targets) == len(outputs)
        cost = 0.0
        for i in range(len(targets)):
            cost += 0.5*(targets[i]-outputs[i])**2
        return cost


########################################################################################################################
# Builder

class Activation(object):
    @staticmethod
    def activate(x):
        raise NotImplementedError("Activation class requires .activate() method to be defined to be defined!")

    @staticmethod
    def derivate(x):
        raise NotImplementedError("Activation class requires .derivate() method to be defined to be defined!")


class TanhActivation(Activation):
    @staticmethod
    def activate(x):
        return math.tanh(x)

    @staticmethod
    def derivate(x):
        return 1 - x*x      # approx for tanh derivative


class SigmoidActivation(Activation):
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivate(x):
        a = SigmoidActivation.activate(x)
        return (1 - a) * a


class SoftplusActivation(Activation):
    @staticmethod
    def activate(x):
        return math.log1p(1 + math.exp(x))

    @staticmethod
    def derivate(x):
        return 1 / (1 + math.exp(-x))


class BinaryActivation(Activation):
    @staticmethod
    def activate(x):
        return 0 if x < 0 else 1

    @staticmethod
    def derivate(x):
        return 0 if x != 0 else math.nan


class ReluActivation(Activation):
    @staticmethod
    def activate(x):
        return 0 if x < 0 else x

    @staticmethod
    def derivate(x):
        return 0 if x < 0 else 1


class LinearActivation(Activation):
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def derivate(x):
        return 1


class Distribution(object):
    @staticmethod
    def apply(neuron):
        raise NotImplementedError('Distribution type class requires .apply method to be implemented!')


class UniformDistribution(Distribution):
    @staticmethod
    def apply(neuron):
        left = 1.0
        for c in range(len(neuron.connections)):
            if c == len(neuron.connections)-1:
                neuron.connections[c].weight = left
                break
            neuron.connections[c].weight = random.uniform(0.00001, left)
            left -= neuron.connections[c].weight
        return neuron


class LacunDristribution(Distribution):
    # @todo LaCun 98
    pass


class Layer(object):
    def __init__(self, neurons=1, activation=None, distribution=None):
        self.neurons      = neurons
        self.distribution = activation if issubclass(activation, Distribution) else distribution
        self.activation   = activation if issubclass(activation, Activation) else None

    def __getitem__(self, item):
        pass


class Builder(object):
    builder  = None
    # activations
    TANH     = TanhActivation
    LINEAR   = LinearActivation
    RELU     = ReluActivation
    SIGMOID  = SigmoidActivation
    SOFTPLUS = SoftplusActivation
    #distributions
    UNIFORM  = UniformDistribution

    def __init__(self):
        self.layers = []

    def __getitem__(self, item):
        return self.layers[item]

    @staticmethod
    def instance():
        if Builder.builder is None:
            Builder.builder = Builder()
        return Builder.builder

    def set(self, item, layer):
        self.layers[item] = layer
        return Builder.builder

    def add(self, layer):
        self.layers.append(layer)
        return Builder.builder

    def compile(self):
        # init empty network
        _nn = Network()
        # 1 input, n hidden, 1 output
        _num_layers = len(self.layers)
        # assert num of layers, MUST be >= 3
        assert _num_layers >= 3
        # add layers
        for l in range(_num_layers):
            assert type(self.layers[l]) is Layer
            # IF last => 0 conn ELSE next layer neuron count (w/o bias)
            _num_connections = 0 if l == _num_layers-1 else self.layers[l+1].neurons
            _num_neurons     = self.layers[l].neurons
            # current layer + layer id for reference
            _layer     = []
            # neurons for the current layer
            for n in range(_num_neurons+1):  # +1 bias
                # create neuron
                _neuron = Neuron(n,l)
                # add connnections
                for _ in range(_num_connections):
                    _connection = Connection()
                    _neuron.connections.append(_connection)
                # apply distribution on neuron weights
                if l < _num_layers-1 and self.layers[l].distribution is not None:
                    _neuron = self.layers[l].distribution.apply(_neuron)

                # setup neuron activation functions
                if l > 0:
                    _neuron.activate = self.layers[l].activation.activate
                    _neuron.derivate = self.layers[l].activation.derivate
                # if bias: output = 1 else: 0
                _neuron.output = 1.0 if n == _num_neurons else _neuron.output
                _layer.append(_neuron)
            # add layer to network's layers
            _nn.layers.append(_layer)
        return _nn


########################################################################################################################
# Testing

nn = Builder.instance().add(Layer(3, Builder.UNIFORM)).add(Layer(6, Builder.RELU, Builder.UNIFORM)).add(Layer(1, Builder.RELU)).compile()
m = 5
for i in range(20000):   # +
    a  = random.randint(0, m)
    b  = random.randint(0, m)
    c  = random.randint(0, m)
    t  = (a * b * c)
    a /= m
    b /= m
    c /= m
    t /= m**3
    o, = nn.train((a, b, c), (t,))
    #print('train  ', (a*m, b*m, c*m), (a*m * b*m * c*m), [o*m**3], ' error ', nn.error)
for _ in range(3):
    a  = random.randint(0, m)/m
    b  = random.randint(0, m)/m
    c  = random.randint(0, m)/m
    o, = nn.predict((a, b, c))
    print('predict', (a*m, b*m, c*m), (a*m * b*m * c*m), [o*m**3])

########################################################################################################################
# Notes

# @note
# hidden layer activation: tanh
# output layer activation: linear(no constraints), logistic(for [0,1]) or exp(strictly positive)

# @note
# bias neurons allow network output to be translated to the desired output range

# @note
# normalization: match inputs to the range to that of the activation function
# denormalize outputs to their original desired state

# @notes
# gpgpu w/ opencl neural network
# meta reinforced learning: use nn to learn the best nn structure
# framework for input normalization/standardization/denormalization

# @notes @relu
# [0,n] float inputs: relu + linear NOT good; the weights, gradients and outputs become too large(+ or -)
# [0,n] float inputs: relu=max(0,x) in the hidden layer prevents weights from blancing out(cause of the 0)

# @notes @sigmoid
# addition: w/ enough layers, iterations and w/ input normalization to [0,1] output will converge to the correct result




