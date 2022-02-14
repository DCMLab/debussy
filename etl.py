import os, re
import pandas as pd
from wavescapes import apply_dft_to_pitch_class_matrix

def apply_dft_to_all(d):
    return {k: apply_dft_to_pitch_class_matrix(v) for k, v in d.items()}

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

def get_pcvs(path, pandas=False):
    _, _, pcv_files = next(os.walk(path))
    pcv_dfs = {get_standard_filename(fname): pd.read_csv(os.path.join(path, fname), sep='\t', index_col=0) for fname in sorted(pcv_files)}
    pcv_dfs = {k: parse_interval_index(v).fillna(0.0) for k, v in pcv_dfs.items()}
    if not pandas:
        pcv_dfs = {k: v.to_numpy() for k, v in pcv_dfs.items()}
    return pcv_dfs