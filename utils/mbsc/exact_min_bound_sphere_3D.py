import numpy as np
from scipy.spatial import ConvexHull

from .fit_sphere_2_points import fit_sphere_2_points


def exact_min_bound_sphere_3D(array):
    """
 Compute exact minimum bounding sphere of a 3D point cloud (or a
 triangular surface mesh) using Welzl's algorithm.

   - X     : M-by-3 list of point co-ordinates or a triangular surface
             mesh specified as a TriRep object.
   - R     : radius of the sphere.
   - C     : 1-by-3 vector specifying the centroid of the sphere.
   - Xb    : subset of X, listing K-by-3 list of point coordinates from
             which R and C were computed. See function titled
             'FitSphere2Points' for more info.

 REREFERENCES:
 [1] Welzl, E. (1991), 'Smallest enclosing disks (balls and ellipsoids)',
     Lecture Notes in Computer Science, Vol. 555, pp. 359-370

 Matlab code author: Anton Semechko (a.semechko@gmail.com)
 Date: Dec.2014"""

    # Get the convex hull of the point set
    hull = ConvexHull(array)
    hull_array = array[hull.vertices]
    hull_array = np.unique(hull_array, axis=0)
    # print(len(hull_array))

    # Randomly permute the point set
    hull_array = np.random.permutation(hull_array)

    if len(hull_array) <= 4:
        R, C = fit_sphere_2_points(hull_array)
        return R, C, hull_array

    elif len(hull_array) < 1000:
        try:
            R, C, _ = B_min_sphere(hull_array, [])

            # Coordiantes of the points used to compute parameters of the
            # minimum bounding sphere
            D = np.sum(np.square(hull_array - C), axis=1)
            idx = np.argsort(D - R ** 2)
            D = D[idx]
            Xb = hull_array[idx[:5]]
            D = D[:5]
            Xb = Xb[D < 1E-6]
            idx = np.argsort(Xb[:, 0])
            Xb = Xb[idx]
            return R, C, Xb
        except:
            raise Exception
    else:
        M = len(hull_array)
        dM = min([M // 4, 300])
        # unnecessary ?
        #		res = M % dM
        #		n = np.ceil(M/dM)
        #		idx = dM * np.ones((1, n))
        #		if res > 0:
        #			idx[-1] = res
        #
        #		if res <= 0.25 * dM:
        #			idx[n-2] = idx[n-2] + idx[n-1]
        #			idx = idx[:-1]
        #			n -= 1

        hull_array = np.array_split(hull_array, dM)
        Xb = np.empty([0, 3])
        for i in range(len(hull_array)):
            R, C, Xi = B_min_sphere(np.vstack([Xb, hull_array[i]]), [])

            # 40 points closest to the sphere
            D = np.abs(np.sqrt(np.sum((Xi - C) ** 2, axis=1)) - R)
            idx = np.argsort(D, axis=0)
            Xb = Xi[idx[:40]]

        D = np.sort(D, axis=0)[:4]
        # print(Xb)
        # print(D)
        # print(np.where(D/R < 1e-3)[0])
        Xb = np.take(Xb, np.where(D / R < 1e-3)[0], axis=0)
        Xb = np.sort(Xb, axis=0)
        # print(Xb)

        return R, C, Xb


def B_min_sphere(P, B):
    eps = 1E-6
    if len(B) == 4 or len(P) == 0:
        R, C = fit_sphere_2_points(B)  # fit sphere to boundary points
        return R, C, P

    # Remove the last (i.e., end) point, p, from the list
    P_new = P[:-1].copy()
    p = P[-1].copy()

    # Check if p is on or inside the bounding sphere. If not, it must be
    # part of the new boundary.
    R, C, P_new = B_min_sphere(P_new, B)
    if np.isnan(R) or np.isinf(R) or R < eps:
        chk = True
    else:
        chk = np.linalg.norm(p - C) > (R + eps)

    if chk:
        if len(B) == 0:
            B = np.array([p])
        else:
            B = np.array(np.insert(B, 0, p, axis=0))
        R, C, _ = B_min_sphere(P_new, B)
        P = np.insert(P_new.copy(), 0, p, axis=0)
    return R, C, P
