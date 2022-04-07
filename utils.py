from functools import lru_cache
from itertools import islice

from matplotlib import pyplot as plt
from wavescapes import legend_decomposition
from wavescapes.color import circular_hue
import numpy as np
import math
from scipy import ndimage
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf

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
    # entropy and np.log have same base e
    utm_entropy = 1 - (entropy(mag_mx, axis=-1) / np.log(mag_mx.shape[-1]))
    utm_entropy = MinMaxScaler().fit_transform(
        utm_entropy.reshape(-1, 1)).reshape(-1)
    if is_square:
        utm_entropy = long2utm(utm_entropy)
    return utm_argmax, utm_max, utm_entropy


def most_resonant2color(max_coeff, opacity, hue_segments=6, **kwargs):
    if hue_segments is None:
        hue_segments = max_coeff.max()
    hue_segment = math.tau / hue_segments
    phase = max_coeff * hue_segment
    mag_dims, phase_dims = opacity.ndim, phase.ndim
    assert mag_dims == phase_dims, f"Both arrays should have the same dimensionality"
    if mag_dims > 1:
        mag_phase_mx = np.dstack([opacity, phase])
    else:
        mag_phase_mx = np.column_stack([opacity, phase])
    return circular_hue(mag_phase_mx, **kwargs)


def make_color_legend(file_path=None):
    """Produce a circular legend for the most_resonant summary wavescapes."""
    def make_pcv(position_of_one):
        return [0] * position_of_one + [1] + [0] * (11 - position_of_one)
    legend = {f'c{i + 1}': (make_pcv(6 - i), [2]) for i in range(0, 6)}
    legend_decomposition(legend, width=5, single_img_coeff=2)
    if file_path is not None:
        plt.savefig(file_path)


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
    all_correlations = np.divide(all_correlations, norm_by, out=np.zeros_like(
        all_correlations), where=norm_by > 0)
    if get_arg_max:
        return np.stack([all_correlations.max(axis=1), all_correlations.argmax(axis=1)]).T
    return all_correlations.max(axis=1)


def pitch_class_matrix_to_tritone(pc_mat):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.

    Args:
        pc_mat (np.array): pitch class matrix

    Returns:
        np.array: matrix of tritone presence
    """
    coeff_nmb = 6
    res = np.linalg.norm(np.multiply(pc_mat/np.linalg.norm(pc_mat, axis=1).reshape(-1, 1),
                                     np.roll(pc_mat/np.linalg.norm(pc_mat,
                                             axis=1).reshape(-1, 1), coeff_nmb, axis=-1)
                                     )[..., :coeff_nmb], axis=-1)
    return res


########################################
# metrics
########################################

def center_of_mass(utm):
    """Computes the vertical center of mass for each coefficient in the wavescape

    Args:
        utm (np.array): array containing the 6 wavescapes

    Returns:
        list: 6 vertical centers of mass (normalized by the height of the wavescape)
    """
    vcoms = []
    shape_y, shape_z = np.shape(utm)[1:3]
    for i in range(shape_z):
        utm_interest = utm[:, :, i]
        vcoms.append(ndimage.measurements.center_of_mass(
            utm_interest)[0] / shape_y)
    return vcoms


def add_to_adj_list(adj_list, a, b):
    """Util function to compute the partition entropy

    Args:
        adj_list (list): list containing the pair of adjacent nodes
        a (int): node 1
        b (int): node 2
    """
    adj_list.setdefault(a, []).append(b)
    adj_list.setdefault(b, []).append(a)


def make_adj_list(max_coeff):
    """Creates an adjacency list from the matrix of most resonant coefficients
       in order to convert the matrix to a network.

    Args:
        max_coeff (np.array): matrix of most resonant coefficients

    Returns:
        list: adjacency list
    """
    adj_list = {}

    utm_index = np.arange(0, max_coeff.shape[0] * max_coeff.shape[1]).reshape(max_coeff.shape[0],
                                                                              max_coeff.shape[1])
    for i in range(len(max_coeff)):
        for j in range(len(max_coeff)):
            if (j < len(max_coeff[i]) - 1) and (max_coeff[i][j] == max_coeff[i][j + 1]):
                add_to_adj_list(adj_list, utm_index[i][j], utm_index[i][j + 1])
            if i < len(max_coeff[i]) - 1:
                for x in range(max(0, j - 1), min(len(max_coeff[i + 1]), j + 2)):
                    if (max_coeff[i][j] == max_coeff[i + 1][x]):
                        add_to_adj_list(
                            adj_list, utm_index[i][j], utm_index[i + 1][x])
    return adj_list


def partititions_entropy(adj_list):
    """Computes the entropy of the different connected components
       of the network obtained from the adjacency list.

    Args:
        adj_list (list): adjacency list

    Returns:
        int: normalized entropy of the partitions
    """
    G = nx.Graph(adj_list)
    components = connected_components(G)
    lengths = [len(x) / G.size() for x in components]
    ent = entropy(lengths) / entropy([1] * G.size())
    return ent


########################################
# utils for metrics
########################################


def make_plots(metadata_metrics, save_name, title, cols,
               figsize=(20, 25), unified=False,
               scatter=False, boxplot=False, ordinal=False, ordinal_col='years_ordinal'):
    """Creates custom plots to show the evolution of the metrics.

    Args:
        metadata_metrics (pd.DataFrame): df containing the already computed metrics
        save_name (str): name used for saving the visualization
        title (str): title of the visualization
        cols (list): list of columns plotted
        figsize (tuple, optional): size of the plot. Defaults to (20,25).
        unified (bool, optional): whether the metrics for each coefficient should be plotted in only one axis. Defaults to False.
        scatter (bool, optional): whether to scatter the points in the unified plot. Defaults to False.
        boxplot (bool, optional): to use boxplots instead of regplots (suggested for ordinal plots). Defaults to False.
        ordinal (bool, optional): whether to show the time evolution as an ordinal number. Defaults to False.
        ordinal_col (str, optional): the column that should be used as ordinal values. Defaults to 'years_ordinal'.
    """

    if unified:
        fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    else:
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        axs = axs.flatten()

    fig.suptitle(title)
    if type(cols) == str:
        len_cols = 2
    else:
        len_cols = len(cols) + 1

    for i in range(1, len_cols):
        if type(cols) == str:
            col = cols
        else:
            if len_cols > 7:
                if i == 1:
                    continue
            col = cols[i-1]

        if ordinal:
            x_col = ordinal_col
        else:
            x_col = 'year'

        if not unified:
            if len_cols > 7:
                ax = axs[i-2]
            else:
                ax = axs[i-1]

            ax.set_title(f"Coefficient {i}")
            if boxplot:
                sns.boxplot(x=x_col, y=col, data=metadata_metrics, ax=ax)
            else:
                sns.regplot(x=x_col, y=col, data=metadata_metrics, ax=ax)
        else:
            if boxplot:
                sns.boxplot(x=x_col, y=col, data=metadata_metrics)
            else:
                sns.regplot(x=x_col, y=col, data=metadata_metrics,
                            scatter=scatter, label=f"Coefficient {i}")
            plt.legend()

    plt.savefig(f'figures/{save_name}.png')


def testing_ols(metadata_matrix, cols, ordinal=False, ordinal_col='years_ordinal'):
    """Function used to test the predictiveness of each measure with respect to 
       the time period.

    Args:
        metadata_matrix (pd.DataFrame): df containing all the metrics
        cols (list): list of columns to be tested
        ordinal (bool, optional): whether to show the time evolution as an ordinal number. Defaults to False.
        ordinal_col (str, optional): the column that should be used as ordinal values. Defaults to 'years_ordinal'.
    """

    scaler = StandardScaler()

    if type(cols) != str:
        metadata_sm = metadata_matrix[metadata_matrix[cols[0]].notnull()]
        metadata_sm[cols] = scaler.fit_transform(metadata_sm[cols])
    else:
        metadata_sm = metadata_matrix[metadata_matrix[cols].notnull()]
        metadata_sm[cols] = scaler.fit_transform(
            np.array(metadata_sm[cols]).reshape(-1, 1))
        cols = [cols]


    metadata_sm = pd.melt(metadata_sm, id_vars=['year', 'last_mc'], value_vars=cols)
    metadata_sm = metadata_sm[~metadata_sm['variable'].str.endswith('2')]
    #print(metadata_sm)
    if ordinal:
        print(ordinal_col) # use ordinal col
    
    results = smf.ols(formula='value ~ year * C(variable) + last_mc + 1 ', data=metadata_sm).fit()
    #for col in cols:
    #    if col[-1] in ['4', '5']:
    #        results = smf.ols(formula=f'{col} ~ year * last_mc + 1 ', data=metadata_sm).fit()
    #        #results = sm.OLS(y, X).fit()
    print('testing results')
    print(results.summary())


def add_to_metrics(metrics_df, dict_metric, name_metrics):
    """Function to add the newly computed metrics to the dataframe.

    Args:
        metrics_df (pd.DataFrame): df containing already computed metrics
        dict_metric (dict): name:metric dictionary
        name_metrics (list/str): list of columns names (str if only one column) 

    Returns:
        _type_: _description_
    """
    if type(name_metrics) == str:
        if name_metrics in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=name_metrics)
        df_tmp = pd.Series(dict_metric, name=name_metrics)
    else:
        if name_metrics[0] in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=name_metrics)
        df_tmp = pd.DataFrame(dict_metric).T
        df_tmp.columns = name_metrics
    metrics_df = metrics_df.merge(df_tmp, left_index=True, right_index=True)
    return metrics_df


########################################
# utils for penta dia classification
########################################


def make_training_set(analyses_df, ninefold_dict, full=True, normalize=True, clean=True, binary=True, extend=False, pcms=False):

    to_df_ext = []

    if full:
        for i in range(len(analyses_df)):
            for j in range(int(analyses_df['to_qb'][i]) - int(analyses_df['from_qb'][i]) - 1):
                for z in range(j):
                    if extend:
                        try:
                            to_df_ext.append(
                                [analyses_df['fname'][i]] +
                                list(ninefold_dict[analyses_df['fname'][i]][squareix2longix(int(analyses_df['to_qb'][i]) - int(analyses_df['from_qb'][i]) - 1 - j, int(analyses_df['to_qb'][i]) - z, int(analyses_df['length_qb'][i]))]) +
                                list(pcms[analyses_df['fname'][i]][squareix2longix(int(analyses_df['to_qb'][i]) - int(analyses_df['from_qb'][i]) - 1 - j, int(analyses_df['to_qb'][i]) - z, int(analyses_df['length_qb'][i]))]) +
                                [analyses_df['structure'][i]]
                            )
                        except Exception as e:
                            print('Not found', e)
                    else:
                        try:
                            to_df_ext.append(
                                [analyses_df['fname'][i]] +
                                list(ninefold_dict[analyses_df['fname'][i]][squareix2longix(int(analyses_df['to_qb'][i]) - int(analyses_df['from_qb'][i]) - 1 - j, int(analyses_df['to_qb'][i]) - z, int(analyses_df['length_qb'][i]))]) +
                                [squareix2longix(int(analyses_df['to_qb'][i]) - int(analyses_df['from_qb'][i]) - 1 - j, int(analyses_df['to_qb'][i]) - z, int(analyses_df['length_qb'][i]))] + 
                                [analyses_df['structure'][i]]
                            )
                        except Exception as e:
                            print('Not found', e)
    else:
        for i in range(len(analyses_df)):
            try:
                to_df_ext.append(
                    [analyses_df['fname'][i]] +
                    list(ninefold_dict[analyses_df['fname'][i]][squareix2longix(int(analyses_df['to_qb'][i]) - int(analyses_df['from_qb'][i]) - 1, int(analyses_df['to_qb'][i]), int(analyses_df['length_qb'][i]))]) +
                    [analyses_df['structure'][i]]
                )
            except Exception as e:
                print('Not found', e)

    if extend:
        ground_truth_train = pd.DataFrame(to_df_ext,
                                        columns=['fname', 'coeff1', 'coeff2', 'coeff3', 'coeff4',
                                                'coeff5', 'coeff6', 'major', 'minor', 'tritone', 
                                                0,1,2,3,4,5,6,7,8,9,10,11,
                                                'structure']
                                        )

    else:
        ground_truth_train = pd.DataFrame(to_df_ext,
                                        columns=['fname', 'coeff1', 'coeff2', 'coeff3', 'coeff4',
                                                'coeff5', 'coeff6', 'major', 'minor', 'tritone', 
                                                'point', 'structure']
                                        )

    if normalize:
        ground_truth_train['tritone'] = ground_truth_train[[
            'tritone']] / ground_truth_train[['tritone']].apply(np.max, axis=0)

    ground_truth_train = ground_truth_train.drop_duplicates()
    ground_truth_train = ground_truth_train[ground_truth_train['structure'].notnull(
    )]

    if clean:
        ground_truth_train = ground_truth_train[ground_truth_train['structure'].isin(
            ['majmin', 'penta', 'octa', 'wt'])]

    if binary:
        ground_truth_train['diatonic'] = [1 if 'majmin' in str(
            x) else 0 for x in ground_truth_train['structure']]
        ground_truth_train['pentatonic'] = [1 if 'penta' in str(
            x) else 0 for x in ground_truth_train['structure']]
        ground_truth_train['octatonic'] = [1 if 'octa' in str(
            x) else 0 for x in ground_truth_train['structure']]
        ground_truth_train['wholetone'] = [1 if 'wt' in str(
            x) else 0 for x in ground_truth_train['structure']]

    return ground_truth_train


def most_resonant_penta_dia(mag_mx, ninefold_mat, clf, add_one=False):
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

        is_long = ninefold_mat.ndim == 2

        if is_long:
            ninefold_mat = long2utm(ninefold_mat)

    to_predict = ninefold_mat[utm_argmax == 4]
    predictions = clf.predict(to_predict)
    utm_argmax[utm_argmax == 4] = [
        6 if pred == 0 else 4 for pred in predictions]

    # entropy and np.log have same base e
    utm_entropy = 1 - (entropy(mag_mx, axis=-1) / np.log(mag_mx.shape[-1]))
    utm_entropy = MinMaxScaler().fit_transform(
        utm_entropy.reshape(-1, 1)).reshape(-1)
    if is_square:
        utm_entropy = long2utm(utm_entropy)
    return utm_argmax, utm_max, utm_entropy
