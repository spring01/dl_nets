import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deep_bm import DBM


with open('digitstrain.txt') as train_txt:
    data_train = np.loadtxt(train_txt, delimiter=',')
    x_train, y_train = data_train[:, :-1], data_train[:, -1]
    y_train = np.array([int(y) for y in y_train])
    bin_x_train = 1.0 * (x_train > 0.5)
    bin_xy_train = (bin_x_train, y_train)

with open('digitsvalid.txt') as valid_txt:
    data_valid = np.loadtxt(valid_txt, delimiter=',')
    x_valid, y_valid = data_valid[:, :-1], data_valid[:, -1]
    y_valid = np.array([int(y) for y in y_valid])
    bin_x_valid = 1.0 * (x_valid > 0.5)
    bin_xy_valid = (bin_x_valid, y_valid)

dbm = DBM(num_feat=bin_x_train.shape[1], num_hid1=100, num_hid2=100,
          max_epoch=1, gibbs_iter=1)
dbm.train_batch(bin_x_train, bin_x_valid)

plt.figure(figsize = (10, 10))
gs = gridspec.GridSpec(10, 10)
gs.update(wspace=0.0, hspace=0.0)
rand = lambda dim: np.random.binomial(1, 0.5, dim)
for i in range(100):
    rand_inp = rand(bin_x_train.shape[1])
    rand_h1 = rand(dbm.num_hid1)
    rand_h2 = rand(dbm.num_hid2)
    (inp_neg, _, _) = dbm.gibbs_sampler((rand_inp, rand_h1, rand_h2), 1000)
    ax = plt.subplot(gs[i])
    ax.set_axis_off()
    plt.imshow(inp_neg.reshape(28, -1), cmap=plt.cm.gray)
plt.show()

