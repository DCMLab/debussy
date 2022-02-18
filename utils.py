from functools import lru_cache
from itertools import islice

from wavescapes.color import circular_hue
import numpy as np
import math
from scipy import ndimage
from scipy.stats import entropy
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler


########################################
# SQUARE <-> LONG matrix transformations
########################################

def utm2long(utm):
    n, m = utm.shape[:2]
    assert n == m, f"Upper triangular matrix is expected to be square, not ({n}, {m})."
    return utm[np.triu_indices(n)]


def longn2squaren(n):
    square_n = np.sqrt(0.25 + 2 * n) - 0.5
    assert square_n % 1. == 0, f"Length {n} does not correspond to an upper triangular matrix in long format."
    return int(square_n)


def long2utm(long):
    n, *m = long.shape
    square_n = longn2squaren(n)
    A = np.zeros_like(long, shape=(square_n, square_n, *m))
    A[np.triu_indices(square_n)] = long
    return A


def longix2squareix(ix, n, from_to=False):
    """ Turn the index of a long format UTM (upper triangle matrix) into
    coordinates of a square format UTM.

    Parameters
    ----------
    ix : int
        Index to convert.
    n : int
        Side length of the square matrix.
    from_to : bool, optional
        By default, the returned coordinates signify (segment_length, last_segment).
        Pass True to return (first_segment, last_segment) instead.


    Returns
    -------
    (int, int)
        See `from_to`.
    """
    x, y = divmod(ix, n)
    if from_to:
        x = y - x
    return x, y


def squareix2longix(x, y, n):
    assert x < n and y < n, "Coordinates need to be smaller than n."
    assert y >= x, f"Coordinates ({x}, {y}) are not within an upper triangular matrix."
    return sum(islice(range(n, -1, -1), x)) + y - x


########################################
# Inspecting complex matrices
########################################

def comp2str(c, dec=2):
    """Interpret a complex number as magnitude and phase and convert into a human-readable string."""
    magn = np.round(abs(c), dec)
    ang = -round(np.angle(c, True)) % 360
    return f"{magn}+{ang}°"

comp2str_vec = np.vectorize(comp2str)

def comp2mag_phase(c, dec=2):
    magn = np.round(abs(c), dec)
    ang = np.round(np.angle(c), dec)
    return magn, ang


def get_coeff(dft, x, y, coeff=None, deg=False, invert_x=False):
    """View magnitude and phase of a particular point in the matrix.

    Parameters
    ----------
    dft : np.array
        (NxNx7) complex square matrix or (Nx7) complex long matrix.
    x : int
        By default, x designates the row of the wavescape ('length-to notation'). If `invert_x` is 
        set to True, x is the leftmost index of the selected interval ('from-to notation').
    y : int
        y-1 is the rightmost index of the selected interval.
    coeff : int, optional
        If you want to look at a single coefficient, pass a number between 0 and 6, otherwise all
        7 will be returned.
    deg : bool, optional
        By default, the complex number will be converted into a string containing the rounded
        magnitude and the angle in degrees. Pass false to get the raw complex number.
    invert_x : bool, optional
        See `x`.

    Returns
    -------
    np.array[str or complex]
        Shape 1 or 7 depending on `coeff`, dtype depends on `deg`.
    """
    assert dft.ndim in (2, 3), f"2D or 3D, not {dft.ndim}D"
    if dft.ndim == 2:
        is_long = True
        long_n, n_coeff = dft.shape
        n = longn2squaren(long_n)
        xs, ys = n, n
    else:
        is_long = False
        xs, ys, n_coeff = dft.shape
    if coeff is not None:
        assert 0 <= coeff < n_coeff, f"0 <= coeff < {n_coeff}"
    assert 0 <= x < xs, f"0 <= x < {xs}; received x = {x}" 
    assert 0 <= y < ys, f"0 <= y < {ys}; received y = {y}"
    if invert_x:
        x = y - x
    if is_long:
        ix = squareix2longix(x, y, n)
        result = dft[ix]
    else:
        result = dft[x, y]
    if coeff is not None:
        result = result[[coeff]]
    if deg:
        return comp2str_vec(result)[:, None]
    return np.apply_along_axis(comp2mag_phase, -1, result).T


########################################
# Summary wavescapes
########################################

def most_resonant(mag_mx, add_one=False):
    """ Inpute: NxNx6 matrix of magnitudes or N(N+1)/2x6 long format
    Computes 3 NxNx1 matrices containing:
        the inverse entropy of the 6 coefficients at each point of the matrix
        the maximum value among the 6 coefficients
        the max coefficient
    """
    is_square = mag_mx.ndim == 3
    utm_max = np.max(mag_mx, axis=-1)
    utm_argmax = np.argmax(mag_mx, axis=-1)
    if add_one:
        utm_argmax = np.triu(utm_argmax + 1)
    if is_square:
        # so we don't apply entropy to zero-vectors
        mag_mx = utm2long(mag_mx)
    utm_entropy = 1 - (entropy(mag_mx, axis=-1) / np.log(mag_mx.shape[-1]))  # entropy and np.log have same base e
    utm_entropy = MinMaxScaler().fit_transform(utm_entropy.reshape(-1, 1)).reshape(-1)
    if is_square:
        utm_entropy = long2utm(utm_entropy)
    return utm_argmax, utm_max, utm_entropy

def most_resonant2color(max_coeff, opacity, **kwargs):
    hue_segment = math.tau / max_coeff.max()
    phase = max_coeff * hue_segment
    mag_dims, phase_dims = opacity.ndim, phase.ndim
    assert mag_dims == phase_dims, f"Both arrays should have the same dimensionality"
    if mag_dims > 1:
        mag_phase_mx = np.dstack([opacity, phase])
    else:
        mag_phase_mx = np.column_stack([opacity, phase])
    return circular_hue(mag_phase_mx, **kwargs)


########################################
# Measures
########################################

PITCH_CLASS_PROFILES = {'mozart_major': [0.20033700035703508,
                                         0.010812613711830977,
                                         0.11399209672667372,
                                         0.012104110819714938,
                                         0.13638736763981654,
                                         0.1226311293358654,
                                         0.018993520966402003,
                                         0.20490464831336042,
                                         0.014611863068643751,
                                         0.07414111247856302,
                                         0.011351150687477073,
                                         0.07973338589461738],
                        'mozart_minor': [0.1889699424221723,
                                         0.008978237532936468,
                                         0.11060533806574259,
                                         0.1308781295283801,
                                         0.011630786793101157,
                                         0.11019324778803881,
                                         0.029590747199631288,
                                         0.21962043820162988,
                                         0.07742468998919953,
                                         0.012129908077215472,
                                         0.020386372564054407,
                                         0.07959216183789804]}

@lru_cache
def get_precomputed_rotations(key):
    assert key in PITCH_CLASS_PROFILES, f"Key needs to be one of {list(PITCH_CLASS_PROFILES.keys())}," \
                                        f"not {key}."
    b = np.array(PITCH_CLASS_PROFILES[key])
    n = b.shape[0]
    B_rotated_cols = np.array([np.roll(b, i) for i in range(n)]).T
    B = B_rotated_cols - b.mean()
    b_std = b.std()
    return B, b_std, n


def max_pearsonr_by_rotation(A, b, get_arg_max=False):
    """ For every row in A return the maximum person correlation from all transpositions of b

    Parameters
    ----------
    A : np.array
      (n,m) matrix where the highest correlation will be found for each row.
    b : np.array or str
      (m,) vector to be rolled m times to find the highest possible correlation.
      You can pass a key to use precomputed values for the profiles contained in
      PITCH_CLASS_PROFILES
    get_arg_max : bool, optional
      By default, an (n,) vector with the maximum correlation per row of A is returned.
      Set to True to retrieve an (n,2) matrix where the second column has the argmax,
      i.e. the number of the rotation producing the highest correlation.

    Returns
    -------
    np.array
      (n,) or (n,2) array
    """
    if isinstance(b, str):
        B, b_std, n = get_precomputed_rotations(b)
    else:
        b = b.flatten()
        n = b.shape[0]
        B_rotated_cols = np.array([np.roll(b, i) for i in range(n)]).T
        B = B_rotated_cols - b.mean()
        b_std = b.std()
    assert n == A.shape[1], f"Profiles in A have length {A.shape[1]} but the profile to roll and" \
                            f"compare has length {n}."
    norm_by = A.std(axis=1, keepdims=True) * b_std * n
    all_correlations = (A - A.mean(axis=1, keepdims=True)) @ B
    all_correlations = np.divide(all_correlations, norm_by, out=np.zeros_like(all_correlations), where=norm_by > 0)
    if get_arg_max:
        return np.stack([all_correlations.max(axis=1), all_correlations.argmax(axis=1)]).T
    return all_correlations.max(axis=1)

    
def center_of_mass(index, utm):
    shape_x, shape_y = np.shape(utm)[:2]
    utm_interest = np.abs(utm[:,:,index])
    vcom, hcom = ndimage.measurements.center_of_mass(utm_interest)
    return (hcom/shape_x, vcom/shape_y)

def pitch_class_matrix_to_tritone(pc_mat):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.
    """
    coeff_nmb = 6
    res = np.linalg.norm(np.multiply(pc_mat, np.roll(pc_mat, coeff_nmb, axis=-1))[...,:coeff_nmb], axis=-1)
    return res


def build_custom_utm_from_one_row(res, how='mean'):
    """
    """
    pcv_nmb = np.shape(res)[0]
    for i in range(1, pcv_nmb):
        for j in range(0, pcv_nmb-i):
            if how == 'mean':
                res[i][i+j] = np.mean(np.array([res[0][i+j],res[i-1][i+j-1]]), axis=0)
            else:
                res[i][i+j] = np.max(np.array([res[0][i+j],res[i-1][i+j-1]]), axis=0)
    return res


########################################
# still to adjust from Digital Musicology project
########################################

def compute_magnitude_entropy(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = np.array([ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec])
    entr = ent.spectral_entropy(sel, 1, method='fft')
    return entr[1:]

def compute_magnitudes(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = np.array([ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec])
    return sel[4]

def compute_peaks(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = [ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec]
    return [len(find_peaks(list(magnitudes))[0]) for magnitudes in sel]

def compute_entropy_phase(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.round(np.angle(apply_dft_to_pitch_class_matrix(arr1)), 2)
    vec = []
    coeffs = []
    height = int(utm.shape[0] * ver_ratio)-1
    sel = np.array(utm[height,:,:]).T
    entr = ent.spectral_entropy(sel, 1, method='fft')
    return entr[1:]

