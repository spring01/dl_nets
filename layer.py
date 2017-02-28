
import numpy as np
from scipy.misc import logsumexp
from misc import sigmoid

class layer_sigmoid(object):
    
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        max_wt = 2 * np.sqrt(6.0 / (dim_in + dim_out))
        self.weight = (np.random.rand(self.dim_in, self.dim_out) - 0.5) * max_wt
        self.bias = np.zeros(self.dim_out)
    
    def forward(self, in_train):
        self.in_train = in_train
        self.out_act = sigmoid(in_train.dot(self.weight) + self.bias)
    
    def backward(self, grad_out):
        self._grad_common(grad_out * self.out_act * (1.0 - self.out_act))
    
    def _grad_common(self, grad_pre_act):
        self.grad_weight = np.outer(self.in_train, grad_pre_act)
        self.grad_bias = grad_pre_act
        self.grad_in = grad_pre_act.dot(self.weight.T)
    

class layer_softmax(layer_sigmoid):
    
    def __init__(self, *args):
        super(layer_softmax, self).__init__(*args)
    
    def forward(self, in_train):
        self.in_train = in_train
        pre_act = in_train.dot(self.weight) + self.bias
        self.out_act = np.exp(pre_act - logsumexp(pre_act))
    
    def backward(self, y_elem):
        grad_pre_act = self.out_act.copy()
        grad_pre_act[y_elem] -= 1.0
        self._grad_common(grad_pre_act)
    
    def loss(self, y_elem):
        return -np.log(self.out_act[y_elem])

