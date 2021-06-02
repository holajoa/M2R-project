import numpy as np
import matplotlib.pyplot as plt


### Thomas process ###

def thomas_cluster_process(
        lambda_parent=10, 
        lambda_daughter=100, 
        radius_cluster=.1, 
        daughter_cov=1, 
        boundaries=[-0.5, 0.5, -0.5, 0.5], 
        plot=True
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

    # K function implementation
    T = np.linspace(0, np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2), 100)
    K = np.pi * T**2 + (1/lambda_parent) * (1 - np.exp(-T**2 / (4*daughter_cov)))
    K_ = (1/lambda_parent) * (1 - np.exp(-T**2 / (4*daughter_cov)))
    
    if plot:
        plt.figure(figsize=(6, 6))
        plt.title('Thomas process', fontsize=14)
        plt.scatter(xx, yy, alpha=0.4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.show()
        
        #plt.subplot(122)
        #plt.title(r'$K(t) - \pi t^2$')
        #plt.plot(T, K_, 'r')
        #plt.legend()

    return np.c_[xx, yy]#, T, K

np.random.seed(7)
tcp = thomas_cluster_process(
        lambda_parent=10, 
        lambda_daughter=50, 
        radius_cluster=.2, 
        daughter_cov=.2, 
        boundaries=[-1, 1, -1, 1], 
        plot=True
    )

