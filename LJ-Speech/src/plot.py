#import scipy.misc
import numpy as np
#for i in range(50, 60):
#    scipy.misc.imsave('../summary/'+str(i+1)+'.jpg', np.load('../summary/temp'+str(i+1)+'.npy').reshape([64, 64]))
from scipy.io.wavfile import read, write
import matplotlib
import librosa.display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('../stdr_1.npy')
g1 = np.load('../stdg_1.npy')
#g2 = np.load('../stdg_2.npy')
#g3 = np.load('../stdg_3.npy')
r2 = np.load('../sstdr_1.npy')
g2 = np.load('../sstdg_1.npy')
l = np.arange(len(r2))
#plt.plot(l, r, label='F(xr)')
#plt.plot(l, g1, label='F(xg), std=0.02')
#plt.plot(l, g2, label='F(xg), std=0.1')
#plt.plot(l, g3, label='F(xg), std=0.5')
plt.plot(l, r2, label='F(xr), std=0.2 after')
plt.plot(l, g2, label='F(xg), std=0.2 after')
plt.ylim((0, 1.1))
plt.legend(loc='upper left')

plt.savefig('../summary/gg.png')

'''
sr, x = read('../samples/Step_4500-1.wav')
sr, y = read('/home/jovyan/LJSpeech-1.1/wavs/LJ001-0010.wav')
x = x.astype(np.float32) / 32768.
y = y.astype(np.float32) / 32768.
print (x, y)
y = y[20000: 20000+len(x)]
y = y / 2 ** 0.5
print (len(x), len(y))
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(x, win_length=None)), ref=np.max), y_axis='linear')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, win_length=None)), ref=np.max), y_axis='linear')
plt.savefig('../summary/ggg.png')
'''

'''
fig, axs = plt.subplot(1, 2)
axs[0].set_xlim([-.5, .5])
axs[1].set_xlim([-.5, .5])
axs[0].set_ylim([0, 2000])
axs[1].set_ylim([0, 2000])
axs[0].hist(x, bins=1000)
axs[1].hist(y, bins=1000)
plt.subplot(2, 1, 1)
librosa.display.waveplot(x, sr=sr, color='red')
plt.subplot(2, 1, 2)
librosa.display.waveplot(y, sr=sr, color='black')
plt.savefig('../summary/ggg.png')
'''
