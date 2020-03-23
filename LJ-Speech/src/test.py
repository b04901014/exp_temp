from utils import *

d = '../../LJSpeech-1.1/wavs'
A = os.listdir(d)
nsamp = 10000
b = [os.path.join(d, x) for x in A if x[-4:] == '.wav'][nsamp+100]
b = readwav(b)[16000: 16000+2**15]
A = [os.path.join(d, x) for x in A if x[-4:] == '.wav'][:2000]

As = []
for i in range(len(A)):
    x = readwav(A[i])[16000: 16000+2**15]
    if len(x) == 2 ** 15:
        As.append(x)
A = np.array(As).transpose()
#b = np.random.randn(A.shape[0]) * b.std()
print (A.shape, b.shape)
proj = np.linalg.lstsq(A, b, rcond=1e-8)
proj, r, _, _ = proj
print (proj)
print (r)
