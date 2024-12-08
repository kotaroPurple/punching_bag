
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from numpy.typing import NDArray


type FindExtremaOutput = tuple[NDArray, NDArray, NDArray, NDArray, NDArray]


def cubic_spline_3pts(x, y, T):
    """
    Apparently scipy.interpolate.interp1d does not support
    cubic spline for less than 4 points.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

    m11, m12, m13 = 2 * _x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3

    M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v = np.array([v1, v2, v3]).T
    k = np.linalg.inv(M).dot(v)

    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1

    t = T[np.r_[T >= x0] & np.r_[T <= x2]]
    t1 = (T[np.r_[T >= x0] & np.r_[T < x1]] - x0) / x1x0
    t2 = (T[np.r_[T >= x1] & np.r_[T <= x2]] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2

    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    q = np.append(q1, q2)

    return t, q


def cubic(X, Y, x):
    spl = CubicSpline(X, Y)
    return spl(x)


def find_extrema_simple(T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
    # Finds indexes of zero-crossings
    S1, S2 = S[:-1], S[1:]
    indzer = np.nonzero(S1 * S2 < 0)[0]
    if np.any(S == 0):
        indz = np.nonzero(S == 0)[0]
        if np.any(np.diff(indz) == 1):
            zer = S == 0
            dz = np.diff(np.append(np.append(0, zer), 0))
            debz = np.nonzero(dz == 1)[0]
            finz = np.nonzero(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2.0)

        indzer = np.sort(np.append(indzer, indz))

    # Finds local extrema
    d = np.diff(S)
    d1, d2 = d[:-1], d[1:]
    indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
    indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1

    # When two or more points have the same value
    if np.any(d == 0):
        imax, imin = [], []

        bad = d == 0
        dd = np.diff(np.append(np.append(0, bad), 0))
        debs = np.nonzero(dd == 1)[0]
        fins = np.nonzero(dd == -1)[0]
        if debs[0] == 1:
            if len(debs) > 1:
                debs, fins = debs[1:], fins[1:]
            else:
                debs, fins = [], []

        if len(debs) > 0:
            if fins[-1] == len(S) - 1:
                if len(debs) > 1:
                    debs, fins = debs[:-1], fins[:-1]
                else:
                    debs, fins = [], []

        lc = len(debs)
        if lc > 0:
            for k in range(lc):
                if d[debs[k] - 1] > 0:
                    if d[fins[k]] < 0:
                        imax.append(np.round((fins[k] + debs[k]) / 2.0))
                else:
                    if d[fins[k]] > 0:
                        imin.append(np.round((fins[k] + debs[k]) / 2.0))

        if len(imax) > 0:
            indmax = indmax.tolist()
            for x in imax:
                indmax.append(int(x))
            indmax.sort()

        if len(imin) > 0:
            indmin = indmin.tolist()
            for x in imin:
                indmin.append(int(x))
            indmin.sort()

    local_max_pos = T[indmax]
    local_max_val = S[indmax]
    local_min_pos = T[indmin]
    local_min_val = S[indmin]

    return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer


def prepare_points_simple(
    T: NDArray, S: NDArray, max_pos: NDArray, max_val: NDArray|None,
    min_pos: NDArray, min_val: NDArray|None, nbsym) -> tuple[NDArray, NDArray]:
    """
    Performs mirroring on signal which extrema can be indexed on
    the position array.

    See :meth:`EMD.prepare_points`.
    """

    # Find indexes of pass
    ind_min = min_pos.astype(int)
    ind_max = max_pos.astype(int)

    # Local variables
    end_min, end_max = len(min_pos), len(max_pos)

    ####################################
    # Left bound - mirror nbsym points to the left
    if ind_max[0] < ind_min[0]:
        if S[0] > S[ind_min[0]]:
            lmax = ind_max[1 : min(end_max, nbsym + 1)][::-1]
            lmin = ind_min[0 : min(end_min, nbsym + 0)][::-1]
            lsym = ind_max[0]
        else:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            lmin = np.append(ind_min[0 : min(end_min, nbsym - 1)][::-1], 0)
            lsym = 0
    else:
        if S[0] < S[ind_max[0]]:
            lmax = ind_max[0 : min(end_max, nbsym + 0)][::-1]
            lmin = ind_min[1 : min(end_min, nbsym + 1)][::-1]
            lsym = ind_min[0]
        else:
            lmax = np.append(ind_max[0 : min(end_max, nbsym - 1)][::-1], 0)
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]
            lsym = 0

    ####################################
    # Right bound - mirror nbsym points to the right
    if ind_max[-1] < ind_min[-1]:
        if S[-1] < S[ind_max[-1]]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = ind_min[max(end_min - nbsym - 1, 0) : -1][::-1]
            rsym = ind_min[-1]
        else:
            rmax = np.append(ind_max[max(end_max - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = len(S) - 1
    else:
        if S[-1] > S[ind_min[-1]]:
            rmax = ind_max[max(end_max - nbsym - 1, 0) : -1][::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = ind_max[-1]
        else:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = np.append(ind_min[max(end_min - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rsym = len(S) - 1

    # In case any array missing
    if not lmin.size:
        lmin = ind_min
    if not rmin.size:
        rmin = ind_min
    if not lmax.size:
        lmax = ind_max
    if not rmax.size:
        rmax = ind_max

    # Mirror points
    tlmin = 2 * T[lsym] - T[lmin]
    tlmax = 2 * T[lsym] - T[lmax]
    trmin = 2 * T[rsym] - T[rmin]
    trmax = 2 * T[rsym] - T[rmax]

    # If mirrored points are not outside passed time range.
    if tlmin[0] > T[0] or tlmax[0] > T[0]:
        if lsym == ind_max[0]:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
        else:
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]

        if lsym == 0:
            raise Exception("Left edge BUG")

        lsym = 0
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]

    if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
        if rsym == ind_max[-1]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
        else:
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]

        if rsym == len(S) - 1:
            raise Exception("Right edge BUG")

        rsym = len(S) - 1
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]

    zlmax = S[lmax]
    zlmin = S[lmin]
    zrmax = S[rmax]
    zrmin = S[rmin]

    tmin = np.append(tlmin, np.append(T[ind_min], trmin))
    tmax = np.append(tlmax, np.append(T[ind_max], trmax))
    zmin = np.append(zlmin, np.append(S[ind_min], zrmin))
    zmax = np.append(zlmax, np.append(S[ind_max], zrmax))

    max_extrema = np.array([tmax, zmax])
    min_extrema = np.array([tmin, zmin])

    # Make double sure, that each extremum is significant
    max_dup_idx = np.where(max_extrema[0, 1:] == max_extrema[0, :-1])
    max_extrema = np.delete(max_extrema, max_dup_idx, axis=1)
    min_dup_idx = np.where(min_extrema[0, 1:] == min_extrema[0, :-1])
    min_extrema = np.delete(min_extrema, min_dup_idx, axis=1)

    return max_extrema, min_extrema


def apply_spline(t: NDArray, extrema: NDArray) -> tuple[NDArray, NDArray]:
    if extrema.shape[1] > 3:
        return t, cubic(extrema[0], extrema[1], t)
    else:
        return cubic_spline_3pts(extrema[0], extrema[1], t)


def extract_local_mean(t: NDArray, data: NDArray, nbsym: int) -> NDArray:
    local_max_pos, local_max_val, local_min_pos, local_min_val, indzer = find_extrema_simple(t, data)
    max_extrema, min_extrema = prepare_points_simple(t, data, local_max_pos, local_max_val, local_min_pos, local_min_val, nbsym=2)
    _, max_spline = apply_spline(t, max_extrema)
    _, min_spline = apply_spline(t, min_extrema)
    return (max_spline + min_spline) / 2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.linspace(0., 4., 100)
    data = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 2 * t)
    t = np.arange(len(t))  # should normalize
    local_max_pos, local_max_val, local_min_pos, local_min_val, indzer = find_extrema_simple(t, data)
    max_extrema, min_extrema = prepare_points_simple(t, data, local_max_pos, local_max_val, local_min_pos, local_min_val, nbsym=2)
    _, max_spline = apply_spline(t, max_extrema)
    _, min_spline = apply_spline(t, min_extrema)
    local_mean = extract_local_mean(t, data, 2)
    #
    plt.plot(t, data)
    # plt.plot(t, max_spline, c='gray')
    # plt.plot(t, min_spline, c='gray')
    plt.plot(t, local_mean, c='pink')
    plt.scatter(local_max_pos, local_max_val, c='C1')
    plt.scatter(local_min_pos, local_min_val, c='C2')
    plt.scatter(max_extrema[0], max_extrema[1], c='red')
    plt.scatter(min_extrema[0], min_extrema[1], c='black')
    plt.show()

