#!/usr/bin/python3

import sys
import random
import numpy as np
from scipy import optimize
import datetime
import math
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QDesktopWidget, QMainWindow, QAction, \
                            QGridLayout, QSizePolicy, QSpacerItem, QFormLayout, QLineEdit, QLabel, QComboBox
from PyQt5.QtCore import QThread, Qt, pyqtSignal as QtSignal, QObject, QCoreApplication
import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rc('font', family='Consolas', size=9)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


########################################################################################################################
# Input & Data Normalization


class Standardizer(object):
    pass


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


class Network(object):
    eta   = 0.15
    alpha = 0.5

    def __init__(self):
        self.layers    = []
        self.error     = 0.0
        self.average   = 0.0
        self.smoothing = 100.0
        self.outputs   = []

    def train(self, inputs, targets, batch_size=1):
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

class LReluActivation(Activation):
    @staticmethod
    def activate(x):
        return 0.01*x if x < 0 else x

    @staticmethod
    def derivate(x):
        return 0.01 if x < 0 else 1


class LinearActivation(Activation):
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def derivate(x):
        return 1


class Distribution(object):
    pass


class UniformDistribution(object):
    # np.random.randn()
    pass


class LacunDristribution(object):
    # @todo LaCun 98
    pass


class Layer(object):
    def __init__(self, neurons=1, activation=None):
        self.neurons    = neurons
        self.activation = activation

    def __getitem__(self, item):
        pass


class Builder(object):
    builder  = None
    TANH     = TanhActivation
    LINEAR   = LinearActivation
    RELU     = ReluActivation
    LRELU    = LReluActivation
    SIGMOID  = SigmoidActivation
    SOFTPLUS = SoftplusActivation

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
                for c in range(_num_connections):
                    _connection = Connection()
                    _neuron.connections.append(_connection)
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
# app|simulation


class DateTime(object):
    datetime = datetime.datetime(2016, 9, 3)

    @staticmethod
    def add(days=1):
        DateTime.datetime = DateTime.datetime + datetime.timedelta(days=days)

    @staticmethod
    def day(dif=None):
        if dif is not None:
            yday = DateTime.datetime + datetime.timedelta(days=dif)
            return yday.day
        else:
            return DateTime.datetime.day


class Client(object):
    def __init__(self):
        self.payrate = random.randrange(1, 10) # 1 poor, 10 rich


class Product(object):
    availabilities = ['In Stock','Limited','Supplier']
    categories     = ['GPU','MainBoard','CPU','HDD','RAM','Keyboard','Mouse']
    pid = 1
    def __init__(self, store):
        self.store        = store
        # properties
        self.id           = Product.pid; Product.pid += 1
        self.category     = random.choice(Product.categories)
        self.name         = random.choice(['Cheap','Pricey','Ugly','Cool']) + ' ' + random.choice(['Intel','Samsung','Apple','Google','NoName']) + ' ' + self.category + ' #' + str(random.randrange(1,100))
        self.base         = math.floor(random.random() * random.randrange(50, 1000)*100)/100
        self.price        = math.floor((self.base + 0.15 * self.base + random.random() * 0.1 * self.base)*100)/100
        self.discount     = 0.0 # 10% of price
        self.availability = random.randrange(0,3)   # in stock, out of stoc, limited
        self.rating       = random.randrange(0,5)
        self.reviews      = random.randrange(0,24)
        # profit = reward
        self.profit       = [0.0] * 32
        self.sales        = [0] * 32
        self.history      = [0.0] * 32


class Order(object):
    def __init__(self, product):
        self.product = product


class Store(object):
    def __init__(self):
        print('Store.__init__()')
        n = random.randrange(20, 50)
        self.products = [None] * n
        for i in range(n):
            self.products[i] = Product(self)
        self.orders = [list] * 32


class State(object):
    size = 6
    def __init__(self, day, price, discount, availability, rating, reviews):
        self.inputs = []
        self.inputs.append(day/31)
        self.inputs.append(price/1000)
        self.inputs.append(discount)
        self.inputs.append(availability/3)
        self.inputs.append(rating/5)
        self.inputs.append(reviews/10)


class Action(object):
    UP   = +1
    KEEP = 0
    DOWN = -1
    actions = [+1, 0, -1]
    def __init__(self, action, product):
        self.action  = Action.actions[action]
        self.product = product


class Environment(object):
    def __init__(self, simulation):
        print('Environment.__init__()')
        # init store
        self.simulation = simulation
        self.store      = Store()
        self.datetime   = None
        # init clients
        n = random.randrange(3000, 10000)
        self.clients = [None] * n
        for i in range(n):
            client = Client()
            self.clients[i] = client

    # customers buy from store
    def evolve(self):
        #print('Environment.evolve()')
        # new day
        DateTime.add(days=1)

        self.store.orders[DateTime.day()] = []

        # based on the day of the month a client buys a product # salary days -> more clientss
        for _ in range(self.calcNumClientByDay(DateTime.day())):
            # stats that determine if a client buys a product
            # profit(0.7), discount(0.2), availability(0.05), rating(0.1), reviews(0.05)
            # richer clients buy more expesive products
            # some categories are better sold than others
            client = random.choice(self.clients)
            prob1 = self.calcBuyProbByPayrate(client.payrate)
            for product in random.sample(self.store.products, 20):
                prob2 = (self.calcBuyProbByProfit(max(0.0, product.price - product.base)) * 0.7 + \
                        self.calcBuyProbByDiscount(product.discount) * 0.2 + \
                        product.availability * 0.05 + \
                        product.rating / 5 * 0.1 + \
                        self.calcBuyProbByReviews(product.reviews) * 0.5) * prob1
                if prob2 > 0.5:
                    self.store.orders[DateTime.day()].append(Order(product))
                    if random.random() < 0.5:
                        break
        for product in self.store.products:
            client = random.choice(self.clients)
            for _ in range(int((self.calcBuyProbByPayrate(client.payrate) + 0.1) * 10)):
                self.store.orders[DateTime.day()].append(Order(product))

    def calcNumClientByDay(self, day):
        # day buy equation
        return max(50+int(25*random.random()), int(len(self.clients) * (0.24*math.sin(day/2.4673+7.4)+0.6))-50+int(50*random.random()))

    def calcBuyProbByProfit(self, price):
        return max(0.0, min(1.0, 8.9/(price+9.0)))

    def calcBuyProbByDiscount(self, discount):
        return max(0.0, min(1.0, 7 * discount**2 / 2))

    def calcBuyProbByReviews(self, reviews):
        return max(0.0, min(1.0, reviews**2/756))

    def calcBuyProbByPayrate(self, payrate):
        return min(1.0, 0.4 + payrate**2 / 112)

    def state(self, product):
        day = DateTime.day()
        return State(day, product.price, product.discount, product.availability, product.rating, product.reviews)

    def exec(self, action):
        action.product.price += 0.06 * action.product.base * action.action

    def reward(self, product):
        t = 0.0
        for order in self.store.orders[DateTime.day()]:
            if order.product.id == product.id:
                t += order.product.price - order.product.base
        return t


class Memory(object):
    def __init__(self, simulation, size=31):
        self.simulation = simulation
        self.size       = size
        self.memory     = [None] * size
        self.cursor     = 0

    def push(self, memory):
        #print(self.cursor, self.size)
        self.memory[self.cursor] = memory
        self.cursor = 0 if self.cursor == self.size-1 else self.cursor + 1

    def sample(self, size):
        return random.sample(self.memory, size)


class Simulation(QThread):
    def __init__(self, app):
        print('Simulation.__init__()')
        QThread.__init__(self)
        self.eps         = 0.98
        self.gamma       = 0.975
        self.app         = app
        self.environment = Environment(self)
        self.memory      = Memory(self)
        self.network     = Builder.instance().add(Layer(State.size)).add(Layer(12, Builder.LRELU)).add(Layer(12, Builder.LRELU)).add(Layer(3, Builder.LINEAR)).compile()

    def __del__(self):
        print('Simulation.__del__()')
        self.wait()

    def run(self):
        print('Simulation.run()')

        # pre-train
        print('Starting.',end='')
        i = 0
        while i < 31:
            # for each product
            cstate = self.environment.state(self.app.product).inputs
            # network -> predict
            qvals = self.network.predict(cstate)
            if random.random() < self.eps: # 98% of the time
                action = random.randrange(0,3)
            else:
                action = np.argmax(np.array(qvals))
            # network -> take action
            self.environment.exec(Action(action, self.app.product))
            self.environment.evolve()
            nstate = self.environment.state(self.app.product).inputs
            reward = self.environment.reward(self.app.product)
            # start remembering
            self.memory.push((cstate, action, reward, nstate))
            i += 1
            print('.',end='',flush=True)

        print('Simulation...')
        i = 0
        while True:
            cstate = self.environment.state(self.app.product).inputs
            # network -> predict
            qvals = self.network.predict(cstate)
            # choose action
            if random.random() < self.eps: # 98% of the time
                action = random.randrange(0,3)
            else:
                action = np.argmax(np.array(qvals))
            # network -> take action
            self.environment.exec(Action(action, self.app.product))

            # this runs for all products - a nn is required for each product
            self.environment.evolve()
            # observe new state & reward
            nstate = self.environment.state(self.app.product).inputs
            reward = self.environment.reward(self.app.product)
            # remember what you did and the results
            self.memory.push((cstate, action, reward, nstate))
            batch = self.memory.sample(1)
            for memory in batch:
                ostate, action, reward, nstate = memory
                oqvals  = self.network.predict(ostate)
                nqvals  = self.network.predict(nstate)
                maxqval = np.max(np.array(nqvals))
                targets = oqvals
                targets[action] = reward + (self.gamma * maxqval)
                #avg = (targets[0] + targets[1] + targets[2])/3
                #targets = [t/avg for t in targets]
                outputs = self.network.train(ostate, targets)
                print(ostate, outputs, targets)
            # has gained more experience
            if self.eps > 0.1:
                self.eps -= 1/1000

            # update price history after env evolve
            for product in self.environment.store.products:
                product.history[DateTime.day()] = product.price
            # data for graph update
            o = 0
            t = 0.0
            for order in self.environment.store.orders[DateTime.day()]:
                if order.product.id == self.app.product.id:
                    o += 1
                t += order.product.price - order.product.base
            data = [{'graph': 'orders', 'x': DateTime.day(), 'y': o},
                    {'graph': 'price',  'x': DateTime.day(), 'y': self.app.product.price},
                    {'graph': 'profit', 'x': DateTime.day(), 'y': self.app.product.price - self.app.product.base},
                    {'graph': 'total',  'x': DateTime.day(), 'y': t},
                    {'graph': 'error',  'x': DateTime.day(), 'y': self.network.error}]
            # update gui
            self.app.update(data)

            #self.sleep(1)
            i += 1
        # get reward(profit)

        # self.environment.evolve()
        # self.app.update()

        # observe new state
        # update network


########################################################################################################################
# gui


class MplCanvas(FigureCanvas):
    # MplCanvas ~ QWidget + FigureCanvasAgg
    def __init__(self, parent=None, width=6, height=4, dpi=100, title=''):
        fig = Figure(figsize=(width,height),dpi=dpi)
        self.title = title
        #fig.suptitle(title)
        # color
        col = QtGui.QPalette().window().color()
        fig.set_facecolor((col.redF(), col.greenF(), col.blueF()))
        # axes
        self.axes = fig.add_subplot(111,axisbg='#FFFFFF')
        # axes need clearing when .plot() is called
        self.axes.hold(False)
        self.axes.set_ylabel(self.title)
        # init canvas
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class StaticMplCanvas(MplCanvas):
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)


class DynamicMplCanvas(MplCanvas):
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.xy = [0 for _ in range(1,31+2)]

    def plot(self, x, y):
        self.xy[x] = y
        self.axes.plot([i for i in range(1,31+2)], self.xy, 'k', alpha=0.7, fillstyle='full')
        self.axes.set_xlim(1, 31)
        self.axes.set_ylabel(self.title)
        self.draw()

    def clear(self):
        self.xy = [0 for _ in range(1, 31+2)]


class Window(QMainWindow):
    def __init__(self, app):
        print('Window.__init__()')
        QMainWindow.__init__(self, flags=Qt.Tool | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)
        self.app = app
        self.setWindowTitle("Skynet")
        self.setGeometry(300, 0, 900, 1024)
        self.widget   = QWidget(self)
        self.widget.setStyleSheet("background-color: #EDEDED")

        layout        = QGridLayout(self.widget)
        self.orders   = DynamicMplCanvas(self.widget, title='order')
        self.profit   = DynamicMplCanvas(self.widget, title='profit')
        self.price    = DynamicMplCanvas(self.widget, title='price')
        #self.features = DynamicMplCanvas(self.widget, title='features')
        self.total    = DynamicMplCanvas(self.widget, title='total profit')
        self.error    = DynamicMplCanvas(self.widget, title='error')

        layout.addWidget(self.orders,   0, 0)
        layout.addWidget(self.profit,   1, 0)
        layout.addWidget(self.price,    2, 0)
        #layout.addWidget(self.features, 3, 0)
        layout.addWidget(self.error,    4, 0)
        layout.addWidget(self.total,    5, 0)

        self.widget.setFocus()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        QCoreApplication.instance().quit()

    def update(self, data):
        print('Window.update()')
        for row in data:
            if 'graph' in row :
                if row['graph'] == 'orders':
                    self.orders.plot(row['x'], row['y'])
                elif row['graph'] == 'price':
                    self.price.plot(row['x'], row['y'])
                elif row['graph'] == 'profit':
                    self.profit.plot(row['x'], row['y'])
                elif row['graph'] == 'total':
                    self.total.plot(row['x'], row['y'])
                elif row['graph'] == 'error':
                    self.error.plot(row['x'], row['y'])

    def clear(self):
        self.orders.clear()
        self.price.clear()
        self.profit.clear()
        # not total
        # self.features.clear()
        self.error.clear()


class Widget(QWidget):
    def __init__(self, app):
        print('Widget.__init__()')
        QWidget.__init__(self, flags=Qt.Tool | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.app = app
        self.setWindowTitle(' ')
        self.setGeometry(300+900, 0, 300, 280)
        self.setStyleSheet("background-color: #EDEDED; font-size: 16px")

        layout = QFormLayout()
        layout.setVerticalSpacing(10)

        self.products = QComboBox()
        for product in self.app.simulation.environment.store.products:
            self.products.addItem(product.name)
        self.products.activated.connect(self.select)
        layout.addRow(self.products)

        product = self.app.simulation.environment.store.products[0]
        self.app.product = product

        self.base  = QLabel('Price(' + str(product.base) +')')
        self.price = QLineEdit()
        self.price.setStyleSheet("background-color: #FFF")
        self.price.setText(str(product.price))
        self.price.setReadOnly(True)
        layout.addRow(self.base, self.price)

        self.discount = QLineEdit()
        self.discount.setStyleSheet("background-color: #FFF")
        self.discount.setText(str(product.discount))
        self.discount.setReadOnly(True)
        layout.addRow(QLabel("Discount"), self.discount)

        self.category = QLabel('GSM')
        self.category.setText(str(product.category))
        layout.addRow(QLabel("Category"), self.category)

        #self.btn2.clicked.connect(self.getdiscount)

        self.availability = QLabel('In Stock')
        self.availability.setText(str(Product.availabilities[product.availability]))
        layout.addRow(QLabel("Availability"), self.availability)

        self.reviews = QLabel('24')
        self.reviews.setText(str(product.reviews))
        layout.addRow(QLabel("Reviews"), self.reviews)

        self.rating = QLabel('4')
        self.rating.setText(str(product.rating))
        layout.addRow(QLabel("Rating"), self.rating)

        self.submit = QPushButton('Change')
        layout.addRow(self.submit)
        self.submit.clicked.connect(self.change)

        self.setLayout(layout)

    def select(self, i):
        self.app.product = self.app.simulation.environment.store.products[i]
        self.base.setText('Price('+str(self.app.product.base)+')')
        self.price.setText(str(self.app.product.price))
        self.discount.setText(str(self.app.product.discount))
        self.category.setText(str(self.app.product.category))
        self.availability.setText(str(Product.availabilities[self.app.product.availability]))
        self.reviews.setText(str(self.app.product.reviews))
        self.rating.setText(str(self.app.product.rating))
        self.app.window.clear()
        print('Widget.select()', i)

    def change(self):
        print('Widget.change()')

    def update(self, data):
        print('Widget.update()')


class Application(QApplication):
    def __init__(self, List, p_str=None):
        print('Application.__init__()')
        super().__init__(List)
        self.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
        self.window     = None
        self.widget     = None
        self.simulation = None
        self.product    = None

    def __del__(self):
        print('Application.__del__()')
        del self.thread

    def run(self):
        print('Application.run()')
        self.simulation = Simulation(app)
        self.window = Window(self)
        self.window.show()
        self.widget = Widget(app)
        self.widget.show()
        self.window.setFocus()
        self.simulation.start()
        sys.exit(app.exec_())

    def update(self, data):
        print('Application.update()')
        self.window.update(data)
        self.widget.update(data)


########################################################################################################################
# main


if __name__ == "__main__":
    app = Application(sys.argv)
    app.run()


########################################################################################################################
# notes



