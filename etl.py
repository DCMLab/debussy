import os, re, gzip
import pandas as pd
import numpy as np
from wavescapes import apply_dft_to_pitch_class_matrix

def apply_dft_to_all(d):
    return {k: apply_dft_to_pitch_class_matrix(v) for k, v in d.items()}


def get_mag_phase_mx(data_folder, normalization='0c', indulge_prototypes=False):
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


def get_pcvs(path, pandas=False):
    _, _, pcv_files = next(os.walk(path))
    pcv_dfs = {get_standard_filename(fname): pd.read_csv(os.path.join(path, fname), sep='\t', index_col=0) for fname in sorted(pcv_files)}
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


def parse_interval_index(df, name='iv'):
    iv_regex = r"\[([0-9]*\.[0-9]+), ([0-9]*\.[0-9]+)\)"
    df = df.copy()
    values = df.index.str.extract(iv_regex).astype(float)
    iix = pd.IntervalIndex.from_arrays(values[0], values[1], closed='left', name=name)
    df.index = iix
    return df