from collections import defaultdict
from functools import lru_cache
import os, re, gzip, json
import pandas as pd
import numpy as np
from wavescapes import apply_dft_to_pitch_class_matrix, build_utm_from_one_row

from utils import most_resonant, pitch_class_matrix_to_tritone, utm2long, long2utm, max_pearsonr_by_rotation
from utils import most_resonant, pitch_class_matrix_to_minor_major,  pitch_class_matrix_to_tritone, center_of_mass, partititions_entropy, make_adj_list

NORM_METHODS = ['0c', 'post_norm', 'max_weighted', 'max']

def get_dfts(debussy_repo='.', long=True):
    pcvs = get_pcvs(debussy_repo)
    return {fname: apply_dft_to_pitch_class_matrix(pcv, long=long) for fname, pcv in pcvs.items()}

def load_pickled_file(path, long=True):
    """Unzips and loads the file and returns it in long or square format."""
    try:
        with gzip.GzipFile(path, "r") as zip_file:
            matrix = np.load(zip_file, allow_pickle=True)
    except Exception as e:
        print(f"Loading pickled {path} failed with exception\n{e}")
        return None
    n, m = matrix.shape[:2]
    if long and matrix.ndim > 2 and n == m:
        matrix = utm2long(matrix)
    if not long and n != m:
        matrix = long2utm(matrix)
    return matrix

def get_mag_phase_mx(data_folder, norm_params, long=True):
    """ Search data_folder for pickled magnitude_phase matrices corresponding to one
    or several normalization methods and load them into a dictionary.

    Parameters
    ----------
    data_folder : str
        Directory to scan for files.
    norm_params : list of tuple
        The return format depends on whether you pass one or several (how, indulge_prototypes) pairs.
    long : bool, optional
        By default, all matrices are loaded in long format. Pass False to cast to square
        matrices where the lower left triangle beneath the diagonal is zero.

    Returns
    -------
    dict of str or dict of dict
        If norm_params is a (list containing a) single tuple, the result is a {debussy_filename -> pickle_filepath}
        dict. If it contains several tuples, the result is a {debussy_filename -> {norm_params -> pickle_filepath}}
    """
    norm_params = check_norm_params(norm_params)
    several = len(norm_params) > 1
    result = defaultdict(dict) if several else dict()
    for norm, fname, path in find_pickles(data_folder, norm_params):
        mag_phase_mx = load_pickled_file(path, long=long)
        if mag_phase_mx is None:
            continue
        if several:
            result[fname][norm] = mag_phase_mx
        else:
            result[fname] = mag_phase_mx
    if len(result) == 0:
        print(f"No pickled numpy matrices with correct file names found in {data_folder}.")
    return dict(result)

def check_norm_params(norm_params):
    """If the argument is a tuple, turn it into a list of one tuple. Then check if
    the tuples correspond to valid normalization parameters."""
    if isinstance(norm_params, tuple):
        norm_params = [norm_params]
    for t in norm_params:
        assert len(t) == 2, f"norm_params need to be (how, indulge_prototypes) pairs, not {t}"
        assert t[0] in NORM_METHODS, f"how needs to be one of {NORM_METHODS}, not {t[0]}"
    return norm_params

def find_pickles(data_folder, norm_params, coeff=None, ext='npy.gz'):
    """ Generator function that scans data_folder for particular filenames
     and yields the paths.

    Parameters
    ----------
    data_folder : str
        Scan the file names in this directory.
    norm_params : list of tuple
        One or several (how, indulge_prototype) pairs.
    coeff : str, optional
        If the filenames include a 'c{N}-' component for coefficient N, select N.
    ext : str, optional
        The extension of the files to detect.

    Yields
    ------
    (str, int), str, str
        For each found file matching the critera, return norm_params, debussy_fname, pickled_filepath
    """
    norm_params = check_norm_params(norm_params)
    data_folder = resolve_dir(data_folder)
    assert os.path.isdir(data_folder), data_folder + " is not an existing directory."
    ext_reg = ext.lstrip('.').replace('.', r'\.') + ')$'
    data_regex = r"^(?P<fname>.*)-"
    if coeff is not None:
        data_regex += r"(?:c(?P<coeff>\d)-)?"
    data_regex += r"(?P<how>0c|post_norm|max|max_weighted)(?P<indulge_prototype>\+indulge)?\.(?P<extension>" + ext_reg
    for f in sorted(os.listdir(data_folder)):
        m = re.search(data_regex, f)
        if m is None:
            continue
        capture_groups = m.groupdict()
        if coeff is not None and str(coeff) != capture_groups['coeff']:
            continue
        does_indulge = capture_groups['indulge_prototype'] is not None
        params = (capture_groups['how'], does_indulge)
        if params in norm_params:
            yield params, capture_groups['fname'], os.path.join(data_folder, f)


def get_correlations(data_folder, long=True):
    """Returns a dictionary of pickled correlation matrices."""
    data_folder = resolve_dir(data_folder)
    result = {}
    for f in sorted(os.listdir(data_folder)):
        if f.endswith('-correlations.npy.gz'):
            fname = f[:-20]
            corr = load_pickled_file(os.path.join(data_folder, f), long=long)
            if corr is not None:
                result[fname] = corr
    if len(result) == 0:
        print(f"No pickled numpy matrices with correct file names found in {data_folder}.")
    return result


def get_maj_min_coeffs(debussy_repo='.', long=True, get_arg_max=False):
    """Returns a dictionary of all pitch-class matrices' maximum correlations with a
    major and a minor profile."""
    pcms = get_pcms(debussy_repo, long=True)
    result = {}
    for fname, pcm in pcms.items():
        maj_min = np.column_stack([
            max_pearsonr_by_rotation(pcm, 'mozart_major', get_arg_max=get_arg_max),
            max_pearsonr_by_rotation(pcm, 'mozart_minor', get_arg_max=get_arg_max)
        ])
        result[fname] = maj_min if long else long2utm(maj_min)
    return result


def get_metadata(debussy_repo='.'):
    md_path = os.path.join(debussy_repo, 'metadata.tsv')
    dur_path = os.path.join(debussy_repo, 'durations/spotify_median_durations.json')
    metadata = pd.read_csv(md_path, sep='\t', index_col=1)
    print(f"Metadata for {metadata.shape[0]} files.")
    with open('durations/spotify_median_durations.json', 'r', encoding='utf-8') as f:
        durations = json.load(f)
    idx2key = pd.Series(metadata.index.str.split('_').map(lambda l: l[0][1:] if l[0] != 'l000' else l[1]), index=metadata.index)
    fname2duration = idx2key.map(durations).rename('median_recording')
    fname2year = ((metadata.composed_end + metadata.composed_start) / 2).rename('year')
    qb_per_minute = (60 * metadata.length_qb_unfolded / fname2duration).rename('qb_per_minute')
    sounding_notes_per_minute = (60 * metadata.all_notes_qb / fname2duration).rename('sounding_notes_per_minute')
    sounding_notes_per_qb = (metadata.all_notes_qb / metadata.length_qb_unfolded).rename('sounding_notes_per_qb')
    return pd.concat([
        metadata,
        fname2year,
        fname2duration,
        qb_per_minute,
        sounding_notes_per_qb,
        sounding_notes_per_minute
    ], axis=1)

def get_most_resonant(mag_phase_mx_dict):
    max_coeff, max_mag, inv_entropy = zip(*(most_resonant(mag_phase_mx[...,0]) 
                                            for mag_phase_mx in mag_phase_mx_dict.values()))
    return (
        dict(zip(mag_phase_mx_dict.keys(), max_coeff)),
        dict(zip(mag_phase_mx_dict.keys(), max_mag)),
        dict(zip(mag_phase_mx_dict.keys(), inv_entropy))
    )

@lru_cache
def get_pcms(debussy_repo='.', long=True):
    pcvs = get_pcvs(debussy_repo, pandas=False)
    return {fname: build_utm_from_one_row(pcv, long=long) for fname, pcv in pcvs.items()}

@lru_cache
def get_pcvs(debussy_repo, pandas=False):
    pcvs_path = os.path.join(debussy_repo, 'pcvs', 'debussy-w0.5-piece-wise-pc-q1-slices-pcvs.tsv')
    pcvs = pd.read_csv(pcvs_path, sep='\t', index_col=[0, 1, 2])
    pcv_dfs = {fname: pcv_df.reset_index(level=[0,1], drop=True) for fname, pcv_df in pcvs.groupby(level=1)}
    if pandas:
        pcv_dfs = {k: parse_interval_index(v) for k, v in pcv_dfs.items()}
    if not pandas:
        pcv_dfs = {k: v.to_numpy() for k, v in pcv_dfs.items()}
    return pcv_dfs


def get_standard_filename(fname):
    fname_filter = r"(l\d{3}(?:-\d{2})?(?:_[a-z]+){1,2})"
    m = re.search(fname_filter, fname)
    if m is None:
        return
    return m.groups(0)[0]

def get_ttms(debussy_repo='.', long=True):
    """Returns a dictionary with the results of the tritone detector run on all pitch-class matrices."""
    pcms = get_pcms(debussy_repo, long=long)
    return {fname: pitch_class_matrix_to_tritone(pcm) for fname, pcm in pcms.items()}


def make_feature_vectors(data_folder, norm_params, long=True):
    """ Return a dictionary with concatenations of magnitude-phase matrices for the
     selected normalizations with the corresponding correlation matrices.

    Parameters
    ----------
    data_folder : str
        Folder containing the pickled matrices.
    norm_params : list of tuple
        The return format depends on whether you pass one or several (how, indulge_prototypes) pairs.
    long : bool, optional
        By default, all matrices are loaded in long format. Pass False to cast to square
        matrices where the lower left triangle beneath the diagonal is zero.

    Returns
    -------
    dict of str or dict of dict
        If norm_params is a (list containing a) single tuple, the result is a {debussy_filename -> feature_matrix}
        dict. If it contains several tuples, the result is a {debussy_filename -> {norm_params -> feature_matrix}}
    """
    norm_params = check_norm_params(norm_params)
    several = len(norm_params) > 1
    result = defaultdict(dict) if several else dict()
    mag_phase_mx_dict = get_mag_phase_mx(data_folder, norm_params, long=True)
    correl_dict = get_correlations(data_folder, long=True)
    m_keys, c_keys = set(mag_phase_mx_dict.keys()), set(correl_dict.keys())
    m_not_c, c_not_m = m_keys.difference(c_keys), c_keys.difference(m_keys)
    if len(m_not_c) > 0:
        print(f"No pickled correlations found for the following magnitude-phase matrices: {m_not_c}.")
    if len(c_not_m) > 0:
        print(f"No pickled magnitude-phase matrices found for the following correlations: {c_not_m}.")
    key_intersection = m_keys.intersection(c_keys)
    for fname in key_intersection:
        corr = correl_dict[fname]
        mag_phase = mag_phase_mx_dict[fname]
        if several:
            for norm in norm_params:
                if not norm in mag_phase:
                    print(f"No pickled magnitude-phase matrix found for the {norm} normalization "
                          f"of {fname}.")
                    continue
                mag_phase_mx = mag_phase[norm][..., 0]
                features = np.column_stack([mag_phase_mx, corr])
                result[fname][norm] = features if long else long2utm(features)
        else:
            features = np.column_stack([mag_phase[..., 0], corr])
            result[fname] = features if long else long2utm(features)
    return result


def parse_interval_index(df, name='iv'):
    iv_regex = r"\[([0-9]*\.[0-9]+), ([0-9]*\.[0-9]+)\)"
    df = df.copy()
    values = df.index.str.extract(iv_regex).astype(float)
    iix = pd.IntervalIndex.from_arrays(values[0], values[1], closed='left', name=name)
    df.index = iix
    return df

def test_dict_keys(dict_keys, metadata):
    found_fnames = metadata.index.isin(dict_keys)
    if found_fnames.all():
        print("Found matrices for all files listed in metadata.tsv.")
    else:
        print(f"Couldn't find matrices for the following files:\n{metadata.index[~found_fnames].to_list()}.")


def resolve_dir(d):
    """ Resolves '~' to HOME directory and turns ``d`` into an absolute path.
    """
    if d is None:
        return None
    if '~' in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)


def get_center_of_mass(mag_phase_mx_dict):
    return {fname: center_of_mass(mag_phase_mx[...,0]) for fname, mag_phase_mx in mag_phase_mx_dict.items()}

def get_mean_resonance(mag_phase_mx_dict):
    return {fname: np.mean(mag_phase_mx[...,0], axis=(0,1)) for fname, mag_phase_mx in mag_phase_mx_dict.items()}

def add_to_metrics(metrics_df, dict_metric, name_metrics):
    if type(name_metrics) == str:
        df_tmp = pd.Series(dict_metric, name=name_metrics)
    else:
        df_tmp = pd.DataFrame(dict_metric).T
        df_tmp.columns = name_metrics
    metrics_df = metrics_df.merge(df_tmp, left_index=True, right_index=True)
    return metrics_df

def get_partition_entropy(max_coeffs):
    return {fname: partititions_entropy(make_adj_list(max_coeff)) for fname, max_coeff in max_coeffs.items()}
    
def get_percentage_resonance(max_coeffs):
    return {fname: np.divide(np.array([(max_coeff == i).sum() for i in range(6)]), max_coeff.shape[0]*max_coeff.shape[1]) for fname, max_coeff in max_coeffs.items()}
    
