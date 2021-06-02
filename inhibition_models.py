import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
