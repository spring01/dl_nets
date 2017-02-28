
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf

class MolTensorNet(object):
    
    _atom_num_dict = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5}
    _num_basis = 30
    _num_hid_rec = 60
    _num_hid = 15
    _inter_mu_vec = np.arange(-1.0, 10.2, 0.2)
    _inter_denom = 2 * 0.2**2
    _num_inter = len(_inter_mu_vec)
    
    def __init__(self, num_recurrent=1, dtype=tf.float64,
                 learning_rate=1e-8, momentum=0.9, max_epoch=100):
        self.num_recurrent = num_recurrent
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epoch = max_epoch
        # initialize initial atom coefficients
        self._coeff_init = self._init_rand_norm((len(self._atom_num_dict), self._num_basis))
        # initialize recurrent weights/biases
        self._weight_cf = self._init_rand((self._num_basis, self._num_hid_rec))
        self._bias_cf = self._init_zero((self._num_hid_rec))
        self._weight_df = self._init_rand((self._num_inter, self._num_hid_rec))
        self._bias_df = self._init_zero((self._num_hid_rec))
        self._weight_fc = self._init_rand((self._num_hid_rec, self._num_basis))
        # initialize fully connected weights/biases
        self._weight_hid = self._init_rand((self._num_basis, self._num_hid))
        self._bias_hid = self._init_zero((self._num_hid))
        self._weight_out = self._init_zero((self._num_hid, 1)) # 0's fine for a linear layer
        self._bias_out = self._init_zero(())
        # tensorflow session and variable initialization
        self._sess = tf.Session()
    
    def init_with_all_data(self, all_cart, all_mol_ener):
        self._compute_atom_ener_trans(all_cart, all_mol_ener)
        max_num_atom = np.max([len(cart) for cart in all_cart])
        self._compute_graph(max_num_atom)
        nbf = self._num_basis
        norm_scale = 1.0 / np.sqrt(nbf)
        self._sess.run(tf.initialize_all_variables())
    
    def format_data(self, all_cart, all_mol_ener):
        data = []
        for cart, mol_ener in zip(all_cart, all_mol_ener):
            z_vec = cart[:, 0]
            dist_mat = squareform(pdist(cart[:, 1:]))
            inter = self._compute_inter(dist_mat)
            one_hot_z = np.zeros((len(z_vec), len(self._atom_num_dict)))
            for ind, z in enumerate(z_vec):
                one_hot_z[ind, self._atom_num_dict[int(z)]] = 1.0
            data += [(len(cart), one_hot_z, inter, mol_ener)]
        return data
    
    def train_sgd(self, formatted_train, formatted_valid):
        format_epoch_risk = 'epoch %d, train risk %f, valid risk %f'
        for epoch in range(self.max_epoch):
            start = time.time()
            np.random.shuffle(formatted_train)
            if epoch % 10 == 0:
                train_risk = self._risk(formatted_train)
                valid_risk = self._risk(formatted_valid)
                #~ valid_risk = 0.0
                print format_epoch_risk % (epoch, train_risk, valid_risk)
            for num, ohz, intr, me in formatted_train:
                grad_graph, one_hot_z, inter, mol_ener = self._grad_graph[num]
                feed_dict = {one_hot_z: ohz, inter: intr, mol_ener: me}
                self._sess.run(grad_graph, feed_dict)
            print '    elapsed %f s' % (time.time() - start)
    
    def pred(self, formatted_data):
        pred = []
        for num, ohz, intr, _ in formatted_data:
            pred_graph, one_hot_z, inter = self._pred_graph[num]
            feed_dict = {one_hot_z: ohz, inter: intr}
            pred += [self._sess.run(pred_graph, feed_dict)]
        return np.array(pred)
    
    def _risk(self, formatted_data):
        loss = 0.0
        for num, ohz, intr, me in formatted_data:
            loss_graph, one_hot_z, inter, mol_ener = self._loss_graph[num]
            feed_dict = {one_hot_z: ohz, inter: intr, mol_ener: me}
            loss += self._sess.run(loss_graph, feed_dict)
        return loss / len(formatted_data)
    
    def _compute_inter(self, dist_mat):
        num_atom = dist_mat.shape[0]
        inter = np.zeros((num_atom, num_atom, len(self._inter_mu_vec)))
        for i in range(num_atom):
            for j in range(i + 1, num_atom):
                temp = (dist_mat[i, j] - self._inter_mu_vec)**2
                inter[i, j, :] = np.exp(-temp / self._inter_denom)
                inter[j, i, :] = inter[i, j, :]
        return inter
    
    def _compute_atom_ener_trans(self, all_cart, all_mol_ener):
        num_atom_list = [cart.shape[0] for cart in all_cart]
        self._atom_ener_mean = sum(all_mol_ener) / sum(num_atom_list)
        zl = zip(all_mol_ener, num_atom_list)
        sum_sq = sum([(me - num * self._atom_ener_mean)**2 for me, num in zl])
        self._atom_ener_std = np.sqrt(sum_sq / sum(num_atom_list))
    
    def _compute_graph(self, max_num_atom):
        self._pred_graph = {num: self._pred_graph_num_atom(num)
                            for num in range(1, max_num_atom + 1)}
        self._loss_graph = {num: self._compute_loss_graph(*lg)
                            for num, lg in self._pred_graph.iteritems()}
        grad = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self._grad_graph = {num: (grad.minimize(lg[0]), lg[1], lg[2], lg[3])
                            for num, lg in self._loss_graph.iteritems()}
    
    def _pred_graph_num_atom(self, num_atom):
        one_hot_z = tf.placeholder(self.dtype, [num_atom, len(self._atom_num_dict)])
        shape_inter = [num_atom, num_atom, self._num_inter]
        inter = tf.placeholder(self.dtype, shape_inter)
        coeff_mat_rec = tf.matmul(one_hot_z, self._coeff_init)
        shape_t4 = [num_atom, num_atom, self._num_hid_rec]
        shape_v = [num_atom, num_atom, self._num_basis]
        v_mask_np = np.zeros((num_atom, num_atom, self._num_basis))
        for i in range(self._num_basis):
            v_mask_np[:, :, i] = 1 - np.eye(num_atom)
        v_mask = tf.constant(v_mask_np, dtype=self.dtype)
        for _ in range(self.num_recurrent):
            temp1 = tf.matmul(coeff_mat_rec, self._weight_cf) + self._bias_cf
            temp2 = tf.reshape(inter, [-1, self._num_inter])
            temp3 = tf.matmul(temp2, self._weight_df) + self._bias_df
            temp4 = tf.reshape(temp3, shape_t4)
            temp5 = tf.reshape(temp1 * temp4, [-1, self._num_hid_rec])
            temp6 = tf.matmul(temp5, self._weight_fc)
            tensor_v = tf.tanh(tf.reshape(temp6, shape_v))
            tensor_v *= v_mask
            coeff_mat_rec += tf.reduce_sum(tensor_v, [1])
        hid_temp1 = tf.matmul(coeff_mat_rec, self._weight_hid) + self._bias_hid
        hid_mat = tf.tanh(hid_temp1)
        norm_atom_ener = tf.matmul(hid_mat, self._weight_out) + self._bias_out
        atom_ener = norm_atom_ener * self._atom_ener_std + self._atom_ener_mean
        mol_ener_pred = tf.reduce_sum(atom_ener)
        return mol_ener_pred, one_hot_z, inter
    
    def _compute_loss_graph(self, mol_ener_pred, one_hot_z, inter):
        mol_ener = tf.placeholder(self.dtype, ())
        loss = tf.square(mol_ener_pred - mol_ener)
        return loss, one_hot_z, inter, mol_ener
    
    def _init_rand(self, shape):
        max_wt = np.sqrt(shape[0])
        unif = tf.random_uniform(shape, -max_wt, max_wt, dtype=self.dtype)
        return tf.Variable(unif)
    
    def _init_zero(self, shape):
        return tf.Variable(tf.zeros(shape, dtype=self.dtype))
    
    def _init_rand_norm(self, shape):
        stddev = 1 / np.sqrt(shape[1])
        normal = tf.random_normal(shape, 0.0, stddev, dtype=self.dtype)
        return tf.Variable(normal)
    


