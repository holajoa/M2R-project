import numpy as np
import matplotlib.pyplot as plt


### RANDOM VS INHIBITED (THINNED) PROCESS ###

def inhibited_process(xmax=11, ymax=11, expectation=20, dist=.4):
    xN, yN = np.random.poisson(expectation*xmax), np.random.poisson(expectation*ymax)
    events = np.c_[np.random.uniform(low=0, high=xmax, size=xN), np.random.uniform(low=0, high=ymax, size=xN)]
    L = len(events)
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            D[i, j] = np.linalg.norm(events[i] - events[j])
    D = D + D.T
    ind = [True] * L
    for i, row in enumerate(D):
        if (row[row>0] < dist).any():
            ind[i] = False
    return events, events[ind]

spp0, spp = inhibited_process()

plt.figure(figsize=(11,5))
plt.subplot(121)
plt.plot(spp0[:, 0], spp0[:, 1], 'go', label='Removed points')
plt.plot(spp[:, 0], spp[:, 1], 'ko', label='Retained points')
plt.xlim(0.5, 10.5)
plt.ylim(0.5, 10.5)
plt.title('Poisson pattern before thinning')
plt.legend()

plt.subplot(122)
plt.plot(spp[:, 0], spp[:, 1], 'ko')
plt.xlim(0.5, 10.5)
plt.ylim(0.5, 10.5)
plt.title('Poisson pattern after Matern I thinning')

plt.show()
