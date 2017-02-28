
import numpy as np
from numpy.random import binomial
from misc import *

class DBM(object):
    
    def __init__(self, num_feat, num_hid1, num_hid2,
                 learning_rate=0.01, max_epoch=200, batch_size=10,
                 gibbs_iter=1, num_chain=100, num_upd_mu=10):
        self.num_feat = num_feat
        self.num_hid1 = num_hid1
        self.num_hid2 = num_hid2
        self.weight1 = init_weight(num_feat, num_hid1)
        self.weight2 = init_weight(num_hid1, num_hid2)
        self.bias_inp = np.zeros(num_feat)
        self.bias_hid1 = np.zeros(num_hid1)
        self.bias_hid2 = np.zeros(num_hid2)
        
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        
        self.gibbs_iter = gibbs_iter
        self.num_chain = num_chain
        self.num_upd_mu = num_upd_mu
    
    def train_batch(self, bin_x_train, bin_x_valid):
        bin_x_train = bin_x_train.copy()
        rand = lambda dim: np.random.binomial(1, 0.5, dim)
        chain = [(rand(self.num_feat), rand(self.num_hid1), rand(self.num_hid2))
                 for _ in range(self.num_chain)]
        all_risk = []
        for epoch in range(self.max_epoch):
            risk_train = self._risk(bin_x_train)
            risk_valid = self._risk(bin_x_valid)
            print epoch, risk_train, risk_valid
            all_risk += [(epoch, risk_train, risk_valid)]
            np.random.shuffle(bin_x_train)
            for mini_batch in batch(bin_x_train, self.batch_size):
                mu_list = [self._get_mu(inp) for inp in mini_batch]
                for ind, sample in enumerate(chain):
                    chain[ind] = self.gibbs_sampler(sample, self.gibbs_iter)
                
                lba = float(len(mini_batch))
                lch = float(len(chain))
                zl = zip(mini_batch, mu_list)
                dir_w1 = sum([np.outer(inp, mu1) for inp, (mu1, _) in zl]) / lba
                dir_w1 -= sum([np.outer(sv, sh1) for sv, sh1, _ in chain]) / lch
                dir_bh1 = sum([mu1 for mu1, _ in mu_list]) / lba
                dir_bh1 -= sum([h1 for _, h1, _ in chain]) / lch
                
                dir_w2 = sum([np.outer(mu1, mu2) for mu1, mu2 in mu_list]) / lba
                dir_w2 -= sum([np.outer(h1, h2) for _, h1, h2 in chain]) / lch
                dir_bh2 = sum([mu2 for _, mu2 in mu_list]) / lba
                dir_bh2 -= sum([h2 for _, _, h2 in chain]) / lch
                
                dir_bi = sum([inp for inp in mini_batch]) / lba
                dir_bi -= sum([v for v, _, _ in chain]) / lch
                
                self.bias_inp += self.learning_rate * dir_bi
                self.weight1 += self.learning_rate * dir_w1
                self.bias_hid1 += self.learning_rate * dir_bh1
                self.weight2 += self.learning_rate * dir_w2
                self.bias_hid2 += self.learning_rate * dir_bh2
        return all_risk
    
    def gibbs_sampler(self, sample, gibbs_iter):
        samp_v, samp_h1, samp_h2 = sample
        samp_v_neg = samp_v
        samp_h2_neg = samp_h2
        for _ in range(gibbs_iter):
            h1_vh2_term1 = samp_v_neg.dot(self.weight1)
            h1_vh2_term2 = self.weight2.dot(samp_h2_neg)
            prob_h1_vh2 = sigmoid(h1_vh2_term1 + h1_vh2_term2 + self.bias_hid1)
            samp_h1_neg = binomial(1, prob_h1_vh2)
            
            prob_h2_h1 = sigmoid(samp_h1_neg.dot(self.weight2) + self.bias_hid2)
            samp_h2_neg = binomial(1, prob_h2_h1)
            
            prob_v_h1 = sigmoid(self.weight1.dot(samp_h1_neg) + self.bias_inp)
            samp_v_neg = binomial(1, prob_v_h1)
        return samp_v_neg, samp_h1_neg, samp_h2_neg
    
    def _get_mu(self, inp):
        mu1 = np.random.rand(self.num_hid1)
        mu2 = np.random.rand(self.num_hid2)
        for _ in range(self.num_upd_mu):
            mu1_term1 = inp.dot(self.weight1)
            mu1_term2 = self.weight2.dot(mu2)
            mu2_term = mu1.dot(self.weight2)
            mu1 = sigmoid(mu1_term1 + mu1_term2 + self.bias_hid1)
            mu2 = sigmoid(mu2_term + self.bias_hid2)
        return mu1, mu2
    
    def _risk(self, bin_x):
        loss = 0.0
        for i, inp in enumerate(bin_x):
            rand_h2 = np.random.binomial(1, 0.5, self.num_hid2)
            sample = (inp, [], rand_h2)
            _, samp_h1_neg, _ = self.gibbs_sampler(sample, self.gibbs_iter)
            act = sigmoid(self.weight1.dot(samp_h1_neg) + self.bias_inp)
            loss -= inp.dot(np.log(act)) + (1.0 - inp).dot(np.log(1.0 - act))
        return loss / bin_x.shape[0]

