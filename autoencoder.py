
import numpy as np
from misc import sigmoid, batch

class autoencoder(object):
    
    def __init__(self, num_feat, num_hid, learning_rate=0.1, momentum=0.0,
                 max_epoch=200, denoise_prob=0.0, batch_size=10):
        max_wt = 2 * np.sqrt(6.0 / (num_feat + num_hid))
        self.weight = (np.random.rand(num_feat, num_hid) - 0.5) * max_wt
        self.bias_enc = np.zeros(num_hid)
        self.bias_dec = np.zeros(num_feat)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epoch = max_epoch
        self.denoise_prob = denoise_prob
        self.batch_size = batch_size
    
    def train_batch(self, bin_x_train, bin_x_valid):
        grad_weight = np.zeros_like(self.weight)
        grad_bias_enc = np.zeros_like(self.bias_enc)
        grad_bias_dec = np.zeros_like(self.bias_dec)
        num_up = 0
        last_risk_valid = np.inf
        all_risk = []
        for epoch in range(self.max_epoch):
            risk_train = self._risk(bin_x_train)
            risk_valid = self._risk(bin_x_valid)
            print epoch, risk_train, risk_valid
            all_risk += [(epoch, risk_train, risk_valid)]
            num_up = num_up + 1 if risk_valid >= last_risk_valid else 0
            last_risk_valid = risk_valid
            if num_up >= 2:
                self.weight += self.learning_rate * grad_weight
                self.bias_enc += self.learning_rate * grad_bias_enc
                self.bias_dec += self.learning_rate * grad_bias_dec
                break
            for mini_batch in batch(bin_x_train, self.batch_size):
                grad_weight *= self.momentum
                grad_bias_enc *= self.momentum
                grad_bias_dec *= self.momentum
                for inp in mini_batch:
                    (masked_inp, hid, rec) = self._forward(inp, train=True)
                    (gw, gb, gc) = self._backward(masked_inp, hid, rec)
                    grad_weight += gw / self.batch_size
                    grad_bias_enc += gb / self.batch_size
                    grad_bias_dec += gc / self.batch_size
                self.weight -= self.learning_rate * grad_weight
                self.bias_enc -= self.learning_rate * grad_bias_enc
                self.bias_dec -= self.learning_rate * grad_bias_dec
        return all_risk
    
    def _forward(self, inp, train):
        if train:
            mask = 1.0 - np.random.binomial(1, self.denoise_prob, len(inp))
        else:
            mask = np.ones(len(inp))
        masked_inp = mask * inp
        hid = sigmoid(masked_inp.dot(self.weight) + self.bias_enc)
        rec = sigmoid(hid.dot(self.weight.T) + self.bias_dec)
        return masked_inp, hid, rec
    
    def _backward(self, masked_inp, hid, rec):
        grad_pre_act_dec = rec - masked_inp
        grad_weight_dec = np.outer(hid, grad_pre_act_dec)
        grad_bias_dec = grad_pre_act_dec
        grad_pre_act_enc = grad_pre_act_dec.dot(self.weight)
        grad_pre_act_enc *= hid * (1.0 - hid)
        grad_weight_enc = np.outer(masked_inp, grad_pre_act_enc)
        grad_bias_enc = grad_pre_act_enc
        grad_weight = grad_weight_enc + grad_weight_dec.T
        return grad_weight, grad_bias_enc, grad_bias_dec
    
    def _risk(self, bin_x):
        loss = 0.0
        for inp in bin_x:
            (minp, _, rec) = self._forward(inp, train=False)
            loss -= minp.dot(np.log(rec)) + (1.0 - minp).dot(np.log(1.0 - rec))
        return loss / bin_x.shape[0]


