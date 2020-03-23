import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', size=10)
import matplotlib.pyplot as plt 

fn = '../expres/results.npy'
sep = 5000
total = 120000
lw = 0.8
results = np.load(fn, allow_pickle=True).item()
#print (results)
#del results['LSGAN_64_nb']
#del results['LSGAN_64_nb_nr']
#del results['LSGAN_64_real']
#results = {'LSGAN_reverse': results['LSGAN_64_nb'], 'LSGAN_nb': results['LSGAN_64_nb_nr'], 'LSGAN': results['LSGAN_64_real']}
results = {'G(z)': results['Proposed_64_2'], 'F(G(z))': results['Proposed_64_2_after_F']}
#print (results)
t = np.arange(total // sep) * sep + sep
for key, value in results.items():
    v, e = np.mean(value, axis=0), np.std(value, axis=0)
    if len(v) != len(t):
        continue
#    plt.errorbar(t, v, yerr=e, label=key, linewidth=lw, fmt='.--', markersize='2', elinewidth=2)
    plt.plot(t, v, '--', label=key, linewidth=lw, markersize='2')
    plt.fill_between(t, v-e, v+e, alpha=0.2, antialiased=True)

plt.xlabel('Iterations')
plt.ylabel('FID')
plt.xlim([sep, total])
plt.ylim([40, 600])
plt.legend(loc='upper right')

plt.savefig('../gg.png')
