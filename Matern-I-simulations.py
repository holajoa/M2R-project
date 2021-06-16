import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import squareform, pdist

def k_estimate(x, y, window=[0, 1, 0, 1], n=200, r_max=None):
    points = np.column_stack((x, y))
    x_min, x_max, y_min, y_max = window
    area = (y_max-y_min)*(x_max-x_min)
    N = len(points)
    lam_hat = N/area
    
    u = squareform(pdist(points)) + np.eye(N)
    d1 = np.minimum(x - x_min, x_max - x)
    d2 = np.minimum(y - y_min, y_max - y)
    d1 = np.tile(d1, (N, 1))
    d2 = np.tile(d2, (N, 1))
    d_hypot = np.hypot(d1, d2)
    
    w1 = 1 - 1/np.pi*(np.arccos(np.minimum(d1, u)/u) + np.arccos(np.minimum(d2, u)/u))
    uu = u.copy()
    uu[uu < d_hypot] = d_hypot[uu < d_hypot]
    w2 = 3/4 - 1/(2*np.pi)*(np.arccos(d1/uu) + np.arccos(d2/uu))
    
    d_hypot = np.hypot(d1, d2)
    w_ind = u <= d_hypot
    w = w_ind*w1 + ~w_ind*w2
    u -= np.eye(N)
    
    if not r_max:
        r_max = min(y_max-y_min, x_max-x_min)/2
    
    r = np.linspace(0, r_max, n)
    k = np.zeros(n)
    for i in range(n):
        d_ind = (u < r[i]) & (u > 0)
        k[i] = np.sum((d_ind/w)/N/lam_hat)
    
    return r, k

def csr(lam, window=[0, 1, 0, 1]):
    x_min, x_max, y_min, y_max = window
    area = (y_max-y_min) * (x_max-x_min)
    N = np.random.poisson(lam * area)
    x_list = np.random.uniform(x_min, x_max, N)
    y_list = np.random.uniform(y_min, y_max, N)
    return np.c_[x_list, y_list]


### RANDOM VS INHIBITED (THINNED) PROCESS ###

import numpy as np

def inhibited_process(xmax=1.1, ymax=1.1, expectation=200, dist=.05):
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
    #print(ind[i])
    return events, events[ind]

np.random.seed(777)
spp0, spp = inhibited_process()
spp0 = spp0 - np.array([0.05, 0.05])
spp = spp - np.array([0.05, 0.05])
intensity = spp.shape[0] / 1
hpp = csr(lam=intensity, window=[0, 1.1, 0, 1.1])
hpp = np.array([s for s in hpp if (s[0]<=1 and s[0]>=0 and s[1]<=1 and s[1]>=0)])
r1, k1 = k_estimate(spp[:, 0], spp[:, 1], window=[0, 1, 0, 1])
r0, k0 = k_estimate(hpp[:, 0], hpp[:, 1], window=[0, 1, 0, 1])

plt.figure(figsize=(17,5))
plt.subplot(131)
plt.scatter(spp0[:, 0], spp0[:, 1], facecolors='none', edgecolors='k', alpha=0.3, label='removed points')
plt.plot(spp[:, 0], spp[:, 1], 'ko')
plt.xlim(0.05, 1.05)
plt.ylim(0.05, 1.05)
plt.title('CSR pattern after Matern I thinning')
plt.xticks([])
plt.yticks([])
plt.legend()

plt.subplot(132)
plt.scatter(hpp[:, 0], hpp[:, 1], facecolor='k')
plt.xticks([])
plt.yticks([])
plt.title('CSR pattern with same intensity as the thinned pattern')

plt.subplot(133)
plt.plot(r0, k0-np.pi*r0**2, '--', label='CSR') 
plt.plot(r1, k1-np.pi*r1**2, 'r-', label='Matern I') 
plt.plot(r0, r0*0, 'k-.')
plt.ylim(-0.055, 0.055)
plt.title(r'$\hat{K}-\pi r^2$')
plt.legend()
plt.show()


### MATERN I & II PROCESSES ###

plt.close('all')

nsim = 10 ** 3

boundaries=[-1, 1, -1, 1]

# Simulation window parameters
xmin, xmax, ymin, ymax = boundaries

# Parameters
lambda_p = 50
radius_core = 0.1

# Extend simulation windows parameters
r_ext = radius_core
xmin_ext = xmin - r_ext
xmax_ext = xmax + r_ext
ymin_ext = ymin - r_ext
ymax_ext = ymax + r_ext

# rectangle dimensions
x_delta_ext = xmax_ext - xmin_ext
y_delta_ext = ymax_ext - ymin_ext
total_area_ext = x_delta_ext * y_delta_ext

N, N_I, N_II = np.zeros((3, nsim))

for ss in range(nsim):
    # Simulate Poisson point process for the parent
    N_ext = np.random.poisson(total_area_ext * lambda_p)
    
    # x and y coordinates of the parents
    xx_ext = xmin_ext + x_delta_ext * np.random.rand(N_ext)
    yy_ext = ymin_ext + y_delta_ext * np.random.rand(N_ext)
    
    # thin points if outside the simulation window
    bool_inside = ((xx_ext >= xmin) & (xx_ext <= xmax) & 
                    (yy_ext >= ymin) & (yy_ext <= ymax))
    idx_window = np.arange(N_ext)[bool_inside]
    
    # retain points inside simulation window
    xx_poisson = xx_ext[bool_inside]
    yy_poisson = yy_ext[bool_inside]
    
    # number of Poisson points in window
    N_poisson = len(xx_poisson)
    
    # create random makes for ages
    mark_age = np.random.rand(N_ext)
    
    ### --------------Start thinning points--------------
    bool_remove_I, bool_keep_II = np.zeros((2, N_poisson), dtype=bool)
    
    for ii in range(N_poisson):
        dist = np.hypot(xx_poisson[ii] - xx_ext, yy_poisson[ii] - yy_ext)
        bool_in_disc = (dist < radius_core) & (dist > 0)  # check if inside disc
        
        # Matern I
        bool_remove_I[ii] = any(bool_in_disc)
        
        # Matern II - keep younger points
        if len(mark_age[bool_in_disc]) == 0:
            bool_keep_II[ii] = True
        else:
            bool_keep_II[ii] = all(mark_age[idx_window[ii]] < mark_age[bool_in_disc])
    
    ### ---------------End thinning points---------------
    
    # Remove/keep points to generate Matern hard-core processes
    # Matérn I
    bool_keep_I = ~(bool_remove_I)
    xxMI = xx_poisson[bool_keep_I]
    yyMI = yy_poisson[bool_keep_I]
    
    # Matérn II
    xxMII = xx_poisson[bool_keep_II]
    yyMII = yy_poisson[bool_keep_II]
    
    # Update statistics
    N[ss] = N_poisson
    N_I[ss] = len(xxMI)
    N_II[ss] = len(xxMII)

matern = plt.figure(constrained_layout=True, figsize=(12, 8))
gs = matern.add_gridspec(2, 3)
markersize = 12
matern_0 = matern.add_subplot(gs[:, :-1])
matern_0.plot(xx_poisson,yy_poisson, 'ko', markerfacecolor="None", markersize=markersize);
matern_0.plot(xxMI, yyMI, 'rx', markersize=markersize/2)
matern_0.plot(xxMII, yyMII, 'b+', markersize=markersize)
matern_0.legend(('Underlying Poisson', 'Matern I', 'Matern II'), fontsize=14)

matern_1 = matern.add_subplot(gs[0, -1])
matern_1.plot(xxMI, yyMI, 'ro')
matern_1.set_title('Matern I')

matern_2 = matern.add_subplot(gs[1, -1])
matern_2.plot(xxMII, yyMII, 'o')
matern_2.set_title('Matern II')

plt.show()
