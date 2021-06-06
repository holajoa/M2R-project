import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

### Functions for estimating K and L function ###

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


def l_estimate(x, y, **kwargs):
    r, k = k_estimate(x, y, **kwargs)
    l = np.sqrt(k/np.pi)
    return r, l



### Thomas process VS CSR ###

def thomas_cluster_process(
        lambda_parent=10, 
        lambda_daughter=100, 
        radius_cluster=.1, 
        daughter_cov=1, 
        boundaries=[-0.5, 0.5, -0.5, 0.5]
    ):
    
    """
    Simulating a Thomas cluster process. 

    ----------------------------------------------------------------
    Input: 
    lambda_parent: intensity of parent events
    lambda_daughter: intensity of child events
    radius_cluster: the radius of the disc centred at parent 
                    events containing all child events
    daughter_cov: variance of dispersion of the child events
    boundaries: four boundary lines of the rectangular study region. 
    plot: if True, produce a scatter plot of the generated pattern. 


    Returns:
    X: A 2D array, containing the x-y coordinates of the child points. 
    """
    
    # Simulation window parameters
    xmin, xmax, ymin, ymax = boundaries
    
    # Extend simulation windows parameters
    r_ext = radius_cluster
    xmin_ext = xmin - r_ext
    xmax_ext = xmax + r_ext
    ymin_ext = ymin - r_ext
    ymax_ext = ymax + r_ext

    # rectangle dimensions
    x_delta_ext = xmax_ext - xmin_ext
    y_delta_ext = ymax_ext - ymin_ext
    total_area_ext = x_delta_ext * y_delta_ext

    # Simulate PPP for the parents
    N_parent = np.random.poisson(total_area_ext * lambda_parent)
    
    # x and y coordinates of Poisson points for the parent
    xxparent = xmin_ext + x_delta_ext * np.random.uniform(0, 1, N_parent)
    yyparent = ymin_ext + y_delta_ext * np.random.uniform(0, 1, N_parent)

    # Simulate PPP for the daughters (ie final point process)
    N_daughter = np.random.poisson(lambda_daughter, N_parent)
    N = N_daughter.sum()
    
    # Generate the (relative) locations by simulating bivariate normal r.v.s
    xx0, yy0 = radius_cluster * np.random.multivariate_normal([0, 0], daughter_cov*np.eye(2), N).T

    # replicate parent points (ie centres of disks/clusters)
    xx = np.repeat(xxparent, N_daughter)
    yy = np.repeat(yyparent, N_daughter)

    # translate points (ie parents points are the centres of cluster disks)
    xx = xx + xx0
    yy = yy + yy0

    # thin points if outside the simulation window
    bool_inside = ((xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax))
    xx = xx[bool_inside]
    yy = yy[bool_inside]
    
    return np.c_[xx, yy]

def csr(lam, window=[0, 1, 0, 1]):
    x_min, x_max, y_min, y_max = window
    area = (y_max-y_min) * (x_max-x_min)
    N = np.random.poisson(lam * area)
    x_list = np.random.uniform(x_min, x_max, N)
    y_list = np.random.uniform(y_min, y_max, N)
    return np.c_[x_list, y_list]

# Plotting
np.random.seed(77)
boundaries = [-1, 1, -1, 1]

tcp = thomas_cluster_process(
        lambda_parent=10, 
        lambda_daughter=50, 
        radius_cluster=.2, 
        daughter_cov=.2, 
        boundaries=boundaries, 
    )

hpp = csr(lam=tcp.shape[0]/4, window=[-1.2, 1.2, -1.2, 1.2])
hpp = np.array([s for s in hpp if (s[0]<=1 and s[0]>=-1 and s[1]<=1 and s[1]>=-1)])

plt.figure(figsize=(23, 6))
plt.subplot(131)
plt.title('Thomas Process')
plt.scatter(tcp[:, 0], tcp[:, 1], alpha=0.4)
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(132)
plt.scatter(hpp[:, 0], hpp[:, 1], alpha=0.4)
plt.title('CSR')
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(133)
r, k = k_estimate(tcp[:, 0], tcp[:, 1], window=boundaries) 
r0, k0 = k_estimate(hpp[:, 0], hpp[:, 1], window=boundaries) 
plt.plot(r, k-np.pi*r**2, 'r-', label='Thomas') 
plt.plot(r0, k0-np.pi*r0**2, '--', label='CSR') 
plt.plot(r, r*0, 'k-.')
plt.title(r'$\hat{K}-\pi r^2$')
plt.legend()
plt.show()
