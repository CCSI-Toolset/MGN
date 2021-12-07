
#adapted from https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py

def get_min_ellipsoid(points, epsilon=1e-3):

    '''
    Returns minimal volume ellipsoid with the Khachiyan algorithm;
    Adapted from https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
    
    points: N x d matrix of n points in R^d
    epsilon: tolerance
    '''

    # \argmax \sum_{j \in N_i} u_{ij}^\intercal S_i u_{ij}
    # s.t. \u_{ij}^{\intercal} S_i u_{ij} \leq 1 \forall j \in N_i


    N, d = np.shape(P)
    d = float(d)

    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initialization
    err = 1.0 + epsilon
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > epsilon:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
        np.dot(P.T, np.dot(np.diag(u), P)) - 
        np.array([[a * b for b in center] for a in center])
    ) / d

    return A