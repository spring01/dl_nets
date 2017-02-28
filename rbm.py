import numpy as np
from numpy.random import binomial
from misc import *

class RBM(object):
    
    def __init__(self, num_feat, num_hid, learning_rate=0.1, max_epoch=100,
                 gibbs_iter=1, batch_size=10, num_chain=100):
        self.weight = init_weight(num_feat, num_hid)
        self.bias_for = np.zeros(num_hid)
        self.bias_rev = np.zeros(num_feat)
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.gibbs_iter = gibbs_iter
        self.batch_size = batch_size
        self.num_chain = num_chain
    
    def train_batch(self, bin_x_train, bin_x_valid):
        rand = lambda dim: np.random.binomial(1, 0.5, dim)
        chain = [(rand(self.weight.shape[0]), rand(self.weight.shape[1]))
                 for _ in range(self.num_chain)]
        all_risk = []
        all_weight_bias = []
        np.random.shuffle(bin_x_train)
        lba = float(self.batch_size)
        lch = float(len(chain))
        for epoch in range(self.max_epoch):
            risk_train = self._risk(bin_x_train)
            risk_valid = self._risk(bin_x_valid)
            print epoch, risk_train, risk_valid
            all_risk += [(epoch, risk_train, risk_valid)]
            weight_bias = (self.weight.copy(),
                           self.bias_for.copy(), self.bias_rev.copy())
            all_weight_bias += [weight_bias]
            for mini_batch in batch(bin_x_train, self.batch_size):
                phgx_batch = [sigmoid(inp.dot(self.weight) + self.bias_for)
                              for inp in mini_batch]
                upd_w = sum([np.outer(inp, phgx)
                             for inp, phgx in zip(mini_batch, phgx_batch)]) / lba
                upd_b = sum([phgx for phgx in phgx_batch]) / lba
                upd_c = sum([inp for inp in mini_batch]) / lba
                for i, neg_samp in enumerate(chain):
                    chain[i] = self.gibbs_sampler(neg_samp[0], self.gibbs_iter)
                upd_w -= sum([np.outer(v, h) for v, h in chain]) / lch
                upd_b -= sum([h for _, h in chain]) / lch
                upd_c -= sum([v for v, _ in chain]) / lch
                self.weight += self.learning_rate * upd_w
                self.bias_for += self.learning_rate * upd_b
                self.bias_rev += self.learning_rate * upd_c
        return all_risk
    
    def gibbs_sampler(self, inp, gibbs_iter):
        prob_hgx_inp = sigmoid(inp.dot(self.weight) + self.bias_for)
        hid_neg = binomial(1, prob_hgx_inp)
        for _ in range(gibbs_iter):
            prob_xgh = sigmoid(hid_neg.dot(self.weight.T) + self.bias_rev)
            inp_neg = binomial(1, prob_xgh)
            prob_hgx = sigmoid(inp_neg.dot(self.weight) + self.bias_for)
            hid_neg = binomial(1, prob_hgx)
        return (inp_neg, hid_neg)
    
    def _risk(self, bin_x):
        loss = 0.0
        for n in range(bin_x.shape[0]):
            inp = bin_x[n]
            (_, hid_neg) = self.gibbs_sampler(inp, gibbs_iter=1)
            act = sigmoid(hid_neg.dot(self.weight.T) + self.bias_rev)
            loss -= inp.dot(np.log(act)) + (1.0 - inp).dot(np.log(1.0 - act))
        return loss / bin_x.shape[0]
