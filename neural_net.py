
import numpy as np
import random
from layer import *
from misc import batch

class neural_net(object):
    
    def __init__(self, arch, learning_rate=0.1, max_epoch=200, momentum=0.0,
                 l2reg=0.0, dropout_prob=0.0, batch_size=10):
        self.layer_list = [layer_sigmoid(arch[index], arch[index + 1])
                           for index in range(len(arch) - 2)]
        self.layer_list += [layer_softmax(arch[-2], arch[-1])]
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.momentum = momentum
        self.l2reg = l2reg
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
    
    def train_batch(self, xy_train, xy_valid):
        num_layer = len(self.layer_list)
        grad_weight = [0.0] * num_layer
        grad_bias = [0.0] * num_layer
        all_risk = []
        for epoch in range(self.max_epoch):
            info = self._report(epoch, xy_train, xy_valid)
            all_risk += [info]
            for mini_batch in batch(zip(*xy_train), self.batch_size):
                grad_weight = [g * self.momentum for g in grad_weight]
                grad_bias = [g * self.momentum for g in grad_bias]
                for x_elem, y_elem in mini_batch:
                    self._forward_prop(x_elem, train=True)
                    self._backward_prop(y_elem)
                    for ind, layer in enumerate(self.layer_list):
                        grad_reg = layer.grad_weight
                        grad_reg += self.l2reg * layer.weight * 2
                        grad_weight[ind] += grad_reg / self.batch_size
                        grad_bias[ind] += layer.grad_bias / self.batch_size
                # update weights and biases
                for ind, layer in enumerate(self.layer_list):
                    layer.weight -= self.learning_rate * grad_weight[ind]
                    layer.bias -= self.learning_rate * grad_bias[ind]
        return all_risk
    
    def _report(self, epoch, xy_train, xy_valid):
        x_train, y_train = xy_train
        x_valid, y_valid = xy_valid
        pred_train, loss_train = self.pred_loss(xy_train)
        pred_valid, loss_valid = self.pred_loss(xy_valid)
        train_risk = np.mean(loss_train)
        train_miss = (pred_train != y_train).sum()
        valid_risk = np.mean(loss_valid)
        valid_miss = (pred_valid != y_valid).sum()
        train_str = 'risk: {:.6f}, miss: {:4d}/{:d}'.format(train_risk,
                                                            train_miss,
                                                            len(y_train))
        valid_str = 'risk: {:.6f}, miss: {:4d}/{:d}'.format(valid_risk,
                                                            valid_miss,
                                                            len(y_valid))
        print '{:4d} train {:s}; valid {:s}'.format(epoch, train_str, valid_str)
        return epoch, train_risk, train_miss, valid_risk, valid_miss
    
    def pred_loss(self, xy):
        pred = np.zeros(len(xy[1]), dtype=int)
        loss = np.zeros(len(xy[1]))
        for index, (x_elem, y_elem) in enumerate(zip(*xy)):
            self._forward_prop(x_elem, train=False)
            pred[index] = self.layer_list[-1].out_act.argmax()
            loss[index] = self.layer_list[-1].loss(y_elem)
        return pred, loss
    
    def _forward_prop(self, x_elem, train):
        in_train = x_elem
        drop_prob = self.dropout_prob
        for layer in self.layer_list[:-1]:
            layer.forward(in_train)
            if train:
                mask = 1 - np.random.binomial(1, drop_prob, len(layer.out_act))
            else:
                mask = (1 - drop_prob) * np.ones(len(layer.out_act))
            layer.out_act *= mask
            in_train = layer.out_act
        self.layer_list[-1].forward(in_train)
    
    def _backward_prop(self, y_elem):
        grad_out = y_elem
        for layer in reversed(self.layer_list):
            layer.backward(grad_out)
            grad_out = layer.grad_in
