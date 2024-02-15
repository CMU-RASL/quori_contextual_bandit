import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
CMAP = cm.get_cmap('PiYG')

x = np.arange(0, 10).astype('int')
# y = 0.4*np.tanh(0.5*((x%10) - 5)) + 0.3
y = np.array([0.001, 0.01, 0.07, 0.15, 0.3, 0.5, 0.65, 0.75, 0.8, 0.8])
print(y - 0.5)
# plt.plot(x, y, '-o')
# plt.axis([0, 10, 0, 1])
# plt.xlabel('Rep Number')
# plt.ylabel('Fatigue Estimate')
# plt.show()

# fig, ax = plt.subplots()
# p = [0.24 , 0.43 , 0.015 ,0.14  ,0.175]
# labels = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4']
# ax.barh(range(5), p, label=labels)
# ax.set_xlim([0, 1])
# plt.show()

# for rep_num in range(10):
#     print(np.arange(rep_num, 200+rep_num, 10))
colors = np.linspace(0, 1, 5)

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
p_vecs = [np.array([0.3, 0.4, 0.5, 0.6, 0.8])]
e_vecs = [np.array([-1.5, -0.5, 0.25, 1.25, 2])]
for p_ind, p in enumerate(p_vecs):
    for e_ind, e in enumerate(e_vecs):
        res = []
        for f in y:
            r = p*(1-f*e)
            res.append(np.minimum(np.array([0.95, 0.95, 0.95, 0.95, 0.95]), np.maximum(np.array([0.05, 0.05, 0.05, 0.05, 0.05]), r)))
        res = np.array(res)
        print(res)
        for action_val in range(5):
            if action_val == 2:
                c = 'gray'
            else:
                c = CMAP(colors[action_val])
            ax[p_ind, e_ind].plot(res[:,action_val], color=c)
        ax[p_ind, e_ind].set_ylim([0,1])
        ax[p_ind, e_ind].set_xticks(np.arange(10), ['Rep\n{}'.format(ii) for ii in range(1, 11)])
# plt.show()