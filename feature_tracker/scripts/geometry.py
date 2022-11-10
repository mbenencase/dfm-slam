import numpy as np



def get_3D_points(p2d, K, depth) -> np.ndarray:
    n_points = len(p2d)

    ones = np.ones((n_points, 1))
    p2d_hom = np.hstack([p2d, ones])

    K_inv = np.linalg.inv(K)

    p3d = (K_inv @ p2d_hom.T).T
    p3d = p3d * depth.reshape(-1, 1)

    return p3d