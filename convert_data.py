
import numpy as np
import scipy.io as sio
import cPickle as pickle

qm7 = sio.loadmat('qm7.mat')

num_mol = len(qm7['R'])

all_cart = []
all_label = []
for ind in range(num_mol):
    source_atom_num = qm7['Z'][ind]
    num_atom = sum(source_atom_num > 0)
    atom_num = np.zeros(num_atom)
    atom_num[...] = source_atom_num[:num_atom]
    source_xyz = qm7['R'][ind]
    xyz = np.zeros((num_atom, 3))
    xyz[...] = source_xyz[:num_atom, :]
    cart = np.array(np.bmat([atom_num.reshape(-1, 1), xyz]))
    label = float(qm7['T'][0, ind])
    all_cart += [cart]
    all_label += [label]
qm7_dict = {'all_cart': all_cart, 'all_mol_ener': all_label}
with open('qm7_dict.pickle', 'w') as pic:
    pickle.dump(qm7_dict, pic, protocol=pickle.HIGHEST_PROTOCOL)
