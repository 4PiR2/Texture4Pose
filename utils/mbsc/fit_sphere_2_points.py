import numpy as np
from scipy import linalg


def fit_sphere_2_points(array):
    """Fit a sphere to a set of 2, 3, or at most 4 points in 3D space. Note that
    point configurations with 3 collinear or 4 coplanar points do not have
    well-defined solutions (i.e., they lie on spheres with inf radius).

    - X     : M-by-3 array of point coordinates, where M<=4.
    - R     : radius of the sphere. R=Inf when the sphere is undefined, as
                specified above.
    - C     : 1-by-3 vector specifying the centroid of the sphere.
                C=nan(1,3) when the sphere is undefined, as specified above.

    Matlab code author: Anton Semechko (a.semechko@gmail.com)
    Date: Dec.2014"""

    N = len(array)

    if N > 4:
        print('Input must a N-by-3 array of point coordinates, with N<=4')
        return

    # Empty set
    elif N == 0:
        R = np.nan
        C = np.full(3, np.nan)
        return R, C

    # A single point
    elif N == 1:
        R = 0.
        C = array[0]
        return R, C

    # Line segment
    elif N == 2:
        R = np.linalg.norm(array[1] - array[0]) / 2
        C = np.mean(array, axis=0)
        return R, C

    else:  # 3 or 4 points
        # Remove duplicate vertices, if there are any
        uniq, index = np.unique(array, axis=0, return_index=True)
        array_nd = uniq[index.argsort()]
        if not np.array_equal(array, array_nd):
            print("found duplicate")
            print(array_nd)
            R, C = fit_sphere_2_points(array_nd)
            return R, C

        tol = 0.01  # collinearity/co-planarity threshold (in degrees)
        if N == 3:
            # Check for collinearity
            D12 = array[1] - array[0]
            D12 = D12 / np.linalg.norm(D12)
            D13 = array[2] - array[0]
            D13 = D13 / np.linalg.norm(D13)

            chk = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
            if np.arccos(chk) / np.pi * 180 < tol:
                R = np.inf
                C = np.full(3, np.nan)
                return R, C

            # Make plane formed by the points parallel with the xy-plane
            n = np.cross(D13, D12)
            n = n / np.linalg.norm(n)
            ##print("n", n)
            r = np.cross(n, np.array([0, 0, 1]))
            r = np.arccos(n[2]) * r / (np.linalg.norm(r) + 1e-9)  # Euler rotation vector
            ##print("r", r)
            Rmat = linalg.expm(np.array([
                [0., -r[2], r[1]],
                [r[2], 0., -r[0]],
                [-r[1], r[0], 0.]
            ]))
            ##print("Rmat", Rmat)
            # Xr = np.transpose(Rmat*np.transpose(array))
            Xr = np.transpose(np.dot(Rmat, np.transpose(array)))
            ##print("Xr", Xr)

            # Circle centroid
            x = Xr[:, :2]
            A = 2 * (x[1:] - np.full(2, x[0]))
            b = np.sum((np.square(x[1:]) - np.square(np.full(2, x[0]))), axis=1)
            C = np.transpose(ldivide(A, b))

            # Circle radius
            R = np.sqrt(np.sum(np.square(x[0] - C)))

            # Rotate centroid back into the original frame of reference
            C = np.append(C, [np.mean(Xr[:, 2])], axis=0)
            C = np.transpose(np.dot(np.transpose(Rmat), C))
            return R, C

        # If we got to this point then we have 4 unique, though possibly co-linear
        # or co-planar points.
        else:
            # Check if the the points are co-linear
            D12 = array[1] - array[0]
            D12 = D12 / np.linalg.norm(D12)
            D13 = array[2] - array[0]
            D13 = D13 / np.linalg.norm(D13)
            D14 = array[3] - array[0]
            D14 = D14 / np.linalg.norm(D14)

            chk1 = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
            chk2 = np.clip(np.abs(np.dot(D12, D14)), 0., 1.)
            if np.arccos(chk1) / np.pi * 180 < tol or np.arccos(chk2) / np.pi * 180 < tol:
                R = np.inf
                C = np.full(3, np.nan)
                return R, C

            # Check if the the points are co-planar
            n1 = np.linalg.norm(np.cross(D12, D13))
            n2 = np.linalg.norm(np.cross(D12, D14))

            chk = np.clip(np.abs(np.dot(n1, n2)), 0., 1.)
            if np.arccos(chk) / np.pi * 180 < tol:
                R = np.inf
                C = np.full(3, np.nan)
                return R, C

            # Centroid of the sphere
            A = 2 * (array[1:] - np.full(len(array) - 1, array[0]))
            b = np.sum((np.square(array[1:]) - np.square(np.full(len(array) - 1, array[0]))), axis=1)
            C = np.transpose(ldivide(A, b))

            # Radius of the sphere
            R = np.sqrt(np.sum(np.square(array[0] - C), axis=0))

            return R, C


def permute_dims(array, dims):
    return np.transpose(np.expand_dims(array, axis=max(dims)), dims)


def ldivide(array, vector):
    return np.linalg.solve(array, vector)
