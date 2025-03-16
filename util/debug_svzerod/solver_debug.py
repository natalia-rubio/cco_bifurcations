import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

E_mat = np.loadtxt("E.txt")
F_mat = np.loadtxt("F.txt")
E_standard_mat = np.loadtxt("E_standard.txt")
F_standard_mat = np.loadtxt("F_standard.txt")

pdb.set_trace()

# ax = sns.heatmap(uniform_data, linewidth=0.5)
# plt.show()

plt.spy(E_mat)
#ax = sns.heatmap(E_mat, linewidth=0.5)
plt.savefig("E.png")

plt.clf()
plt.spy(F_mat)
plt.savefig("F.png")