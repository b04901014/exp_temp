import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', size=10)
import matplotlib.pyplot as plt 

fn = '../expres/results_std.npy'
sep = 1
total = 10000
lw = 1.5
smf = 10
sep *= smf
results = np.load(fn, allow_pickle=True).item()
results = {
    'x': results['Proposed_64_r'],
    '0.5x': results['Proposed_64_r'] * 0.5,
    'k=2, F(x)': results['Proposed_64_Fr'],
    'k=2, F(G(z))': results['Proposed_64_Fg'],
    'k=1, F(x)': results['Proposed_64_k1_Fr'],
    'k=1, F(G(z))': results['Proposed_64_k1_Fg'],
}
t = np.arange(total // sep) * sep + sep
for key, value in results.items():
    value = np.reshape(value, [-1, total // smf, smf])
    value = np.mean(value, axis=2)
    v, e = np.mean(value, axis=0), np.std(value, axis=0)
    if len(v) != len(t):
        continue
#    plt.errorbar(t, v, yerr=e, label=key, linewidth=lw, fmt='.--', markersize='2', elinewidth=2)
    plt.plot(t, v, '--', label=key, linewidth=lw, markersize='5')
    plt.fill_between(t, v-e, v+e, alpha=0.4, antialiased=True)

plt.xlabel('Iterations')
plt.ylabel('Standard Deviation')
plt.xlim([sep, 2500])
plt.ylim([0, 0.8])
plt.legend(loc='upper right')

plt.savefig('../ggg.png')
