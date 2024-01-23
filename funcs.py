import kmedoids
def clustering(pairwise_distances, n_clusters):
    '''Returns a dictionary with n_clusters clusters and each cluster assigned a values of a
    list of animal indexes belonging to it'''
    k_medoids_result = kmedoids.fasterpam(pairwise_distances, n_clusters)
    clusters = {cluster:list(np.where(k_medoids_result.labels==cluster)[0]) for cluster in range(n_clusters)}
    return clusters


def gw_distance_mesh(im1,im2):
    mesh = im1
    X = GM(X=mesh.vertices,Tris=mesh.faces,mode="surface",gauge_mode="djikstra",squared=False)

    mesh = im2
    Y = GM(X=mesh.vertices,Tris=mesh.faces,mode="surface",gauge_mode="djikstra",squared=False)

    #compute GW Plan
    P,log = ot.gromov.gromov_wasserstein(X.g,Y.g,X.xi,Y.xi,log=True)
    return(log["gw_dist"])

def gw_distance_2d(im1,im2):
    points,measure = utils.img2atomic(im1)
    X = GM(mode="euclidean",gauge_mode = "euclidean",X=points,xi=measure,normalize_gauge=True)
    points,measure = utils.img2atomic(im2)
    Y = GM(mode="euclidean",gauge_mode = "euclidean",X=points,xi=measure,normalize_gauge=True)

    #compute GW Plan
    P,log = ot.gromov.gromov_wasserstein(X.g,Y.g,X.xi,Y.xi,log=True)
    return(log["gw_dist"])

def gw_distances(ims):
    dists = np.empty((len(ims),len(ims)))
    for i in range(len(ims)):
        for j in range(len(ims)):
            dists [i,j] = gw_distance(ims[i],ims[j])
    return dists