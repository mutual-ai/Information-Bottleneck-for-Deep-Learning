import numpy as np
import math
import random
import utils


class BNLayer:
    def __init__(self, parent):
        self.parent_layer = parent
        self.next_layer = parent.next_layer
        self.out_dim = self.parent_layer.out_dim
        
        self.beta = np.zeros(self.out_dim)
        self.gamma = np.ones(self.out_dim)
        self.dbeta = np.zeros(self.out_dim)
        self.dgamma = np.zeros(self.out_dim)
        
        
        self.lrate = self.parent_layer.learning_rate
        self.reg = self.parent_layer.reg
        self.momentum = self.parent_layer.momentum
        
    def transform(self, X):
        self.X = X
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        self.normalized_X = (X - self.mu) / np.sqrt(self.var + 1e-10)
        self.vals = self.gamma * self.normalized_X + self.beta
    
    def compute_delta(self):
        self.delta = self.next_layer.delta.dot(self.next_layer.W.T) * self.parent_layer.derivative(self.parent_layer.vals)
        
    def compute_para_delta(self):
        
        N, D = self.X.shape
        self.Xmu = self.X - self.mu
        self.XmuN = self.Xmu/N
        self.std_inv = 1. / np.sqrt(self.var + 1e-10)

        self.d_normalized_X = self.delta * self.gamma

        inv = self.std_inv * self.std_inv * self.std_inv
        self.dvar = - np.sum(self.d_normalized_X * self.Xmu, axis = 0) * (1/2.) * inv
        self.dmuN = 1./N * (np.sum(self.d_normalized_X * -self.std_inv, axis = 0) - self.dvar * (2) * np.mean(self.Xmu, axis = 0))

        self.dX = (self.d_normalized_X * self.std_inv) + 2 * (self.dvar * self.XmuN) + self.dmuN
        self.dgamma = np.sum(self.delta * self.normalized_X, axis=0) + self.momentum*self.dgamma
        self.dbeta = np.sum(self.delta, axis=0) + self.momentum * self.dbeta
        
    def weight_update(self):
        self.bn.para_update()
        self.gamma -= self.lrate*self.dgamma
        self.beta -= self.lrate*self.dbeta 


class LayerArgs:
    def __init__(self, in_dim, out_dim, derivative = utils.d_sigmoid, activate = utils.sigmoid, layer_type = "HIDDEN", learning_rate = 0.05, momentum = 0., regularization = 0.000):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.derivative = derivative
        self.activate = activate
        self.type = layer_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization

class Layer:
    def __init__(self, args):

        self.in_dim = args.in_dim
        self.out_dim = args.out_dim

        self.vals = None

        self.type = args.type
        self.learning_rate = args.learning_rate
        self.reg = args.regularization
        self.momentum = args.momentum
        self.derivative = args.derivative
        self.activate = args.activate

        self.next_layer = None
        self.prev_layer = None

        self.initialize_weights()

    def initialize_weights(self):
        unib = math.sqrt(6)/math.sqrt(self.in_dim + self.out_dim)
        self.W = np.random.uniform(-unib, unib, (self.in_dim, self.out_dim))
        self.b = np.zeros((1, self.out_dim))

        self.dW = np.zeros((self.in_dim, self.out_dim))
        self.db = np.zeros((1, self.out_dim))

    def epoch_size(self):
        return self.vals.shape[0]
    
    def connect_layer(self, next_layer):
        self.next_layer = next_layer
        next_layer.prev_layer = self
        
    def layer_forward(self):
        self.wx = self.prev_layer.vals.dot(self.W) + self.b

        if self.type == "INPUT":
            self.vals = self.activate(self.wx)
        else:
            self.bn.transform(self.wx)
            self.vals = self.activate(self.bn.vals)

        if self.type == "OUTPUT":
            self.prob = self.vals / np.sum(self.vals, axis=1, keepdims=True)

    def epoch_size(self):
        return self.vals.shape[0]

    def layer_backward(self, y = None):

        if self.type == "OUTPUT":
            self.bn.delta = np.copy(self.prob)
            self.bn.delta -= y
            self.bn.compute_para_delta()

        elif self.type == "HIDDEN":
            self.bn.compute_delta()
            self.bn.compute_para_delta()
        
        self.delta = self.bn.dX

        self.dW = (self.prev_layer.vals.T).dot(self.delta) + self.momentum * self.dW
        self.db = np.sum(self.delta, axis=0, keepdims=True) + self.momentum * self.db

        self.W -= self.learning_rate * self.dW /self.epoch_size()
        self.b -= self.learning_rate * self.db /self.epoch_size()
        
    def loss(self, gold):
        logprobs = -np.multiply(gold, np.log(self.prob))
        data_loss = np.sum(logprobs)
        return 1./self.epoch_size() * data_loss
    
    

    def add_bn(self):
        self.bn = BNLayer(self)

class ModelArgs:
    def __init__(self, num_passes = 100, max_iter = 500, batch_size = 20, report_interval = 10):
        self.num_passes = num_passes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.report_interval = report_interval

class Model:
    def __init__(self, layer_args, model_arg):

        self.layer_args = layer_args
        self.max_iter = model_arg.max_iter
        self.num_passes = model_arg.num_passes
        self.batch_size = model_arg.batch_size
        self.report_interval = model_arg.report_interval

    def feed_data(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.input_dim = self.X_train.shape[1]
        self.output_dim = len(self.y_train)
        self.train_log_loss = {}
        self.test_log_loss = {}
        self.train_log_acc = {}
        self.test_log_acc = {}

    def trial_data(self, X_train_sub, y_train_sub):

        self.X_train_sub = X_train_sub
        self.y_train_sub = y_train_sub

    def make_layer(self, args):
        return Layer(args)
        
    def yield_batches(self, features, classes, batchsize):
        sets = np.arange(features.shape[0])
        np.random.shuffle(sets)
        for i in range(0, features.shape[0] - batchsize + 1, batchsize):
            e = sets[i:i + batchsize]
            yield features[e], classes[e]
            
    def intialize_model(self):
        self.input_layer = self.make_layer(self.layer_args[0])
        self.output_layer = self.make_layer(self.layer_args[-1])
        self.hidden_layers = [self.make_layer(self.layer_args[i]) for i in range(1, len(self.layer_args)-1)]

        layers = [self.input_layer] + self.hidden_layers + [self.output_layer]

        for i in range(len(layers)-1):
            layers[i].connect_layer(layers[i+1])
        
        for i in range(1, len(layers)):
            layers[i].add_bn()

    def forward(self, x):
        self.input_layer.vals = x
        for layer in self.hidden_layers:
            layer.layer_forward()
        self.output_layer.layer_forward()

    def loss(self, y):
        return self.output_layer.loss(y)
    
    def backward(self, y):
        self.output_layer.layer_backward(y)

        for layer in self.hidden_layers[::-1]:
            layer.layer_backward()

    def run_model(self):
        n_iter = 0
        for i in range(1, self.num_passes+1):
            for x, y in self.yield_batches(self.X_train, self.y_train, self.batch_size):
                n_iter += 1
                self.forward(x)
                self.backward(y)
                if n_iter%self.report_interval == 0:
                    self.forward(self.X_train_sub)
                    yield n_iter, [layer.vals for layer in self.hidden_layers]

            self.forward(self.X_test)
            self.test_log_loss[i] = self.loss(self.y_test)

            self.forward(self.X_train)
            self.train_log_loss[i] = self.loss(self.y_train)
            print "Epoch: {}, Train Acc: {}, Test Acc: {}".format(i, self.train_log_loss[i], self.test_log_loss[i])

            if n_iter > self.max_iter:
                break
