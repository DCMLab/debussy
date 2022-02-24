from functools import lru_cache
import os, re, gzip, json
import pandas as pd
import numpy as np
from wavescapes import apply_dft_to_pitch_class_matrix, build_utm_from_one_row

from utils import most_resonant, pitch_class_matrix_to_minor_major,  pitch_class_matrix_to_tritone, center_of_mass, partititions_entropy, make_adj_list

def get_dfts(debussy_repo='.'):
    pcvs = get_pcvs(debussy_repo)
    return {fname: apply_dft_to_pitch_class_matrix(pcv) for fname, pcv in pcvs.items()}


def get_mag_phase_mx(data_folder, normalization='0c', indulge_prototypes=False):
    """_summary_

    Parameters
    ----------
    data_folder : _type_
        _description_
    normalization : str, optional
        _description_, by default '0c'
    indulge_prototypes : bool, optional
        _description_, by default False

    Returns
    -------
    dict
        {fname -> np.array} (NxNx6x2) magnitude-phase matrices with selected normalization applied.
    """
    how = ('0c', 'post_norm', 'max_weighted', 'max')
    assert normalization in how, f"normalization needs to be one of {how}, not {normalization}"
    data_folder = os.path.expanduser(data_folder)
    assert os.path.isdir(data_folder), data_folder + " is not an existing directory."
    data_regex = r"^(?P<fname>.*)-(?:c(?P<coeff>\d)-)?(?P<how>0c|post_norm|max|max_weighted)(?P<indulge_prototype>\+indulge)?\.(?P<extension>png|npy\.gz)$"
    # this regex can also be used for the computed wavescapes, which is why it includes the <coeff> group and allows for the extension png
    result = {}
    for f in os.listdir(data_folder):
        m = re.search(data_regex, f)
        if m is not None:
            capture_groups = m.groupdict()
            does_indulge = capture_groups['indulge_prototype'] is not None
            if capture_groups['how'] == normalization and does_indulge == indulge_prototypes:
                path = os.path.join(data_folder, f)
                try:
                    with gzip.GzipFile(path, "r") as zip_file:
                        mag_phase_mx = np.load(zip_file, allow_pickle=True)
                        result[capture_groups['fname']] = mag_phase_mx
                except Exception:
                    print(path)
    if len(result) == 0:
        print(f"No pickled numpy matrices with correct file names found in {data_folder}.")
    return result

def get_maj_min_coeffs(debussy_repo='.'):
    pcms = get_pcms(debussy_repo)
    return {fname: pitch_class_matrix_to_minor_major(pcm) for fname, pcm in pcms.items()}


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
def get_pcms(debussy_repo='.'):
    pcvs = get_pcvs(debussy_repo, pandas=False)
    return {fname: build_utm_from_one_row(pcv) for fname, pcv in pcvs.items()} 

@lru_cache
def get_pcvs(debussy_repo, pandas=False):
    pcv_path = os.path.join(debussy_repo, 'pcvs')
    pcv_files = [f for f in os.listdir(pcv_path) if f.endswith('pcvs.tsv')]
    pcv_dfs = {get_standard_filename(fname): pd.read_csv(os.path.join(pcv_path, fname), sep='\t', index_col=0) for fname in sorted(pcv_files)}
    pcv_dfs = {k: parse_interval_index(v).fillna(0.0) for k, v in pcv_dfs.items()}
    if not pandas:
        pcv_dfs = {k: v.to_numpy() for k, v in pcv_dfs.items()}
    return pcv_dfs


def get_standard_filename(fname):
    fname_filter = r"(l\d{3}(?:-\d{2})?(?:_[a-z]+){1,2})"
    m = re.search(fname_filter, fname)
    if m is None:
        return
    return m.groups(0)[0]

def get_ttms(debussy_repo='.'):
    pcms = get_pcms(debussy_repo)
    return {fname: pitch_class_matrix_to_tritone(pcm) for fname, pcm in pcms.items()}


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

def get_center_of_mass(mag_phase_mx_dict):
    return {fname: center_of_mass(mag_phase_mx[...,0]) for fname, mag_phase_mx in mag_phase_mx_dict.items()}

def get_mean_resonance(mag_phase_mx_dict):
    return {fname: np.mean(mag_phase_mx[...,0], axis=(0,1)) for fname, mag_phase_mx in mag_phase_mx_dict.items()}

def add_to_metrics(metrics_df, dict_metric, name_metrics):
    
    if type(name_metrics) == str:
        df_tmp = pd.Series(dict_metric, name=name_metrics)
    else:
        if name_metrics[0] in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=name_metrics)
        df_tmp = pd.DataFrame(dict_metric).T
        df_tmp.columns = name_metrics
    metrics_df = metrics_df.merge(df_tmp, left_index=True, right_index=True)
    return metrics_df

def get_partition_entropy(max_coeffs):
    return {fname: partititions_entropy(make_adj_list(max_coeff)) for fname, max_coeff in max_coeffs.items()}
    
def get_percentage_resonance(max_coeffs, entropy_mat=False):
    if entropy_mat == False:
        return {fname: np.divide(np.array([(max_coeff == i).sum() for i in range(6)]), max_coeff.shape[0]*max_coeff.shape[1]) for fname, max_coeff in max_coeffs.items()}
    else:
        return {fname: np.divide(np.array([(entropy_mat[fname] * (max_coeff == i)).sum() for i in range(6)]), max_coeff.shape[0]*max_coeff.shape[1]) for fname, max_coeff in max_coeffs.items()}

def get_moment_of_inertia(max_coeffs, max_mags):
    
    
    return {fname: np.divide(np.array(
        [
            (
                max_mags[fname][max_coeff == i] * 
                (np.flip(np.square(np.divide(np.indices(max_mags[fname].shape)[0], max_coeff.shape[1]))))[max_coeff == i]
            
            ).sum() 
        for i in range(6)
        ]), 
        max_coeff.shape[0]*max_coeff.shape[1]) 
        for fname, max_coeff in max_coeffs.items()
        
        }


