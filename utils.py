from wavescapes import build_utm_from_one_row
from wavescapes.color import circular_hue
from glob import glob
import numpy as np
import math
from scipy import ndimage
from scipy.stats import entropy
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from networkx.algorithms.components import connected_components


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



MOZART_PROFILES = {
 'major': {0: 0.20033700035703508,
  1: 0.010812613711830977,
  2: 0.11399209672667372,
  3: 0.012104110819714938,
  4: 0.13638736763981654,
  5: 0.1226311293358654,
  6: 0.018993520966402003,
  7: 0.20490464831336042,
  8: 0.014611863068643751,
  9: 0.07414111247856302,
  10: 0.011351150687477073,
  11: 0.07973338589461738},
 'minor': {0: 0.1889699424221723,
  1: 0.008978237532936468,
  2: 0.11060533806574259,
  3: 0.1308781295283801,
  4: 0.011630786793101157,
  5: 0.11019324778803881,
  6: 0.029590747199631288,
  7: 0.21962043820162988,
  8: 0.07742468998919953,
  9: 0.012129908077215472,
  10: 0.020386372564054407,
  11: 0.07959216183789804}}

key_profiles = {
    'major': np.array(list(MOZART_PROFILES['major'].values())),
    'minor': np.array(list(MOZART_PROFILES['minor'].values()))
}
rotated_kp = {(mode, rotation): np.roll(kp, rotation) for mode, kp in key_profiles.items() for rotation in range(12)}




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
        (NxNx6) complex matrix 
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
    xs, ys, _ = dft.shape
    assert 0 <= x < xs, f"0 <= x < {xs}; received x = {x}" 
    assert 0 <= y < ys, f"0 <= y < {ys}; received y = {y}"
    if invert_x:
        x = y - x
    if coeff is None:
        result = dft[x,y]
    else:
        assert 0 <= coeff < 7, "0 <= coeff < 7"
        result = dft[x,y,[coeff]]
    if deg:
        return comp2str_vec(result)[:, None]
    return np.apply_along_axis(comp2mag_phase, -1, result).T


def most_resonant(mag_mx, add_one=False):
    """ Inpute: NxNx6 matrix of magnitudes
    Computes 3 NxNx1 matrices containing:
        the inverse entropy of the 6 coefficients at each point of the matrix
        the maximum value among the 6 coefficients
        the max coefficient
    """

    utm_max = np.max(mag_mx, axis=2)
    utm_argmax = np.argmax(mag_mx, axis=2)
    if add_one:
        utm_argmax = np.triu(utm_argmax + 1)
    utm_entropy = MinMaxScaler().fit_transform(1 - (entropy(mag_mx, axis=2) / entropy(np.array([1]*mag_mx.shape[2]))))  #the normalization is completely arbitrary and should be changed

    return utm_argmax, utm_max, utm_entropy

def most_resonant2color(max_coeff, opacity, **kwargs):
    hue_segment = math.tau / max_coeff.max()
    phase = max_coeff * hue_segment
    mag_phase_mx = np.dstack([opacity, phase])
    return circular_hue(mag_phase_mx, **kwargs)


    
def center_of_mass(utm):
    vcoms = []
    shape_y, shape_z = np.shape(utm)[1:3]
    for i in range(shape_z):
        utm_interest = utm[:,:,i]
        vcoms.append(ndimage.measurements.center_of_mass(utm_interest)[0]/shape_y)
    return vcoms

def pitch_class_matrix_to_tritone(pc_mat):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.
    """
    coeff_nmb = 6
    res = np.linalg.norm(np.multiply(pc_mat, np.roll(pc_mat, coeff_nmb, axis=2))[...,:coeff_nmb], axis=2)
    return res.reshape((res.shape[0], res.shape[1], 1))


def pitch_class_matrix_to_minor_major(pc_mat, rotated_kp=rotated_kp):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.
    """
    #res = np.linalg.norm(np.apply_along_axis(max_correlation, 2, pc_mat, rotated_kp ), axis=2)
    res = max_correlation(pc_mat, rotated_kp)
    
    return res

def max_correlation(pc_mat, rotated_kp):
    coeffs_major_minor = np.array([[[pearsonr(pc_mat[i,j,:], kp)[0] for kp in rotated_kp.values()] for j in range(pc_mat.shape[1])] for i in range(pc_mat.shape[0])])
    return np.stack((coeffs_major_minor[...,:12].max(axis=2), coeffs_major_minor[...,12:].max(axis=2)), axis=2)


def add_to_adj_list(adj_list, a, b):
    adj_list.setdefault(a, []).append(b)
    adj_list.setdefault(b, []).append(a)

def make_adj_list(max_coeff):
    adj_list = {}
        
    utm_index = np.arange(0, max_coeff.shape[0]*max_coeff.shape[1]).reshape(max_coeff.shape[0], max_coeff.shape[1])
    for i in range(len(max_coeff)):
        for j in range(len(max_coeff)):
            if (j < len(max_coeff[i]) - 1) and (max_coeff[i][j] == max_coeff[i][j+1]):
                add_to_adj_list(adj_list, utm_index[i][j], utm_index[i][j+1])
            if i < len(max_coeff[i]) - 1:
                for x in range(max(0, j - 1), min(len(max_coeff[i+1]), j+2)):
                    if (max_coeff[i][j] == max_coeff[i+1][x]):
                        add_to_adj_list(adj_list, utm_index[i][j], utm_index[i+1][x])
    return adj_list

def partititions_entropy(adj_list):
    G = nx.Graph(adj_list)
    components = connected_components(G)
    lengths = [len(x)/G.size() for x in components]
    ent = entropy(lengths) / entropy([1]*G.size())
    return ent

