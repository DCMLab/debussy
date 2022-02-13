import os, re
import pandas as pd
import json
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

def compute_median_from_tracks_list(filename):
    
    # reading from file 
    pieces_data = json.loads(open(filename, "r").read())

    # obtaining medians 
    median_durations = {}
    for piece in pieces_data:
        durations = [t['duration_ms'] for t in  pieces_data[piece]] 
        if not durations: 
            print(piece + " duration missing")
            median_dur = -1
        else: 
            median_dur = pd.DataFrame(durations).median().values[0]
        median_durations[piece] = median_dur
    return median_durations