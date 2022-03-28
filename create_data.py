import os, argparse
import gzip
from collections import defaultdict
from itertools import product
import pathos.multiprocessing as mp

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from wavescapes import normalize_dft, \
    Wavescape  # pip install https://github.com/DCMLab/wavescapes/archive/refs/heads/johannes.zip
from wavescapes.color import circular_hue
from etl import get_dfts, resolve_dir, get_pcms, get_mag_phase_mx
from utils import pitch_class_matrix_to_tritone, max_pearsonr_by_rotation, most_resonant, \
    most_resonant2color, long2utm


def check_and_create(d):
    """ Turn input into an existing, absolute directory path.
    """
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            if input(d + ' does not exist. Create? (y|n)') == "y":
                os.mkdir(d)
            else:
                raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    return resolve_dir(d)


def check_dir(d):
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    return resolve_dir(d)

def make_filename(fname, how, indulge_prototypes, coeff=None, summary_by_entropy=None, ext=None):
    result = fname
    if coeff is not None:
        result += f"-c{coeff}"
    result += f"-{how}"
    if indulge_prototypes:
        result += "+indulge"
    if summary_by_entropy is not None:
        result += "-summary-by-ent" if summary_by_entropy else "-summary-by-mag"
    if ext is not None:
        result += ext
    return result

def core_msg(cores):
    return "in a for-loop." if cores < 1 else f"using {cores} CPU cores in parallel."

def store_wavescapes(wavescape_folder,
                     data_folder,
                     norm_params,
                     coeffs=None,
                     overwrite_standard=False,
                     overwrite_grey=False,
                     overwrite_summary=False,
                     cores=0,
                     sort=False):
    print("Loading magnitude-phase matrices...", end=' ')
    mag_phase_dict = get_mag_phase_mx(data_folder, norm_params)
    first_norm = norm_params[0]
    if len(norm_params) == 1:
        # unify dict structure
        mag_phase_dict = {k: {first_norm: v} for k, v in mag_phase_dict.items()}
    print("DONE")
    if sort:
        print("Sorting by length...", end=' ')
        mag_phase_dict = dict(sorted(mag_phase_dict.items(), key=lambda t: t[1][first_norm].shape[0]))
        print("DONE")
    if coeffs is None:
        coeffs = list(range(1,7))
    print("Assemble file paths and names based on parameters...", end=' ')
    fpath2params = {
        'standard': {os.path.join(wavescape_folder,
                                  make_filename(k,
                                                how,
                                                indulge,
                                                coeff
                                                )
                                  ): (k, how, indulge, coeff)
                    for k, (how, indulge), coeff in product(mag_phase_dict.keys(),
                                                            norm_params,
                                                            coeffs)
                    if not (indulge and coeff == 6)},
        'summary': {os.path.join(wavescape_folder,
                                 make_filename(k,
                                               how,
                                               indulge,
                                               summary_by_entropy=by_entropy,
                                               ext='.png'
                                               )
                                 ): (k, how, indulge, by_entropy)
                    for k, (how, indulge), by_entropy in product(mag_phase_dict.keys(), norm_params, (False, True))},
    }
    fpath2params['grey'] = {path + '-grey.png': params for path, params in fpath2params['standard'].items()}
    fpath2params['standard'] = {path + '.png': params for path, params in fpath2params['standard'].items()}
    # check for existing files that are not to be overwritten
    ws_types = tuple(fpath2params.keys())
    overwrite = dict(zip(ws_types, (overwrite_standard, overwrite_grey, overwrite_summary)))
    for ws_type in ws_types:
        if not overwrite[ws_type]:
            fpath2params[ws_type] = {path: params
                                     for path, params in fpath2params[ws_type].items()
                                     if not os.path.isfile(path)}
    print("DONE")
    print("Getting settled...")
    key2type2params = defaultdict(lambda: {t: [] for t in ws_types})
    for ws_type, path2params in fpath2params.items():
        for path, params in path2params.items():
            key, *p = params
            key2type2params[key][ws_type].append((path, *p))
    # delete unneeded mag_phase_matrices to save RAM
    delete_keys = [k for k in mag_phase_dict.keys() if k not in key2type2params]
    for k in delete_keys:
        del(mag_phase_dict[k])
    for key, type2params in key2type2params.items():
        delete_keys = []
        for norm in mag_phase_dict[key]:
            if not any(norm == (how, indulge) for params in type2params.values() for
                       _, how, indulge, *_ in params):
                delete_keys.append(norm)
        for norm in delete_keys:
            del(mag_phase_dict[key][norm])
    pieces = len(key2type2params)
    parameters = []
    for key, type2params in key2type2params.items():
        for ws_type, params in type2params.items():
            for p in params:
                if ws_type == 'summary':
                    path, how, indulge, by_entropy = p
                    parameters.append((
                        path,
                        mag_phase_dict[key][(how, indulge)],
                        key,
                        how,
                        indulge,
                        None,
                        False,
                        by_entropy
                    ))
                else:
                    path, how, indulge, coeff = p
                    grey = ws_type == 'grey'
                    parameters.append((
                        path,
                        mag_phase_dict[key][(how, indulge)],
                        key,
                        how,
                        indulge,
                        coeff,
                        grey,
                        False
                    ))
    n = len(parameters)
    print(f"Computing {n} wavescapes for {pieces} pieces {core_msg(cores)}...")
    _ = do_it(make_wavescape, parameters, n=n, cores=cores)


def make_wavescape(path, mag_phase_mx, fname, how, indulge, coeff=None, grey=False, by_entropy=False):
    label = f"{fname}: "
    if coeff is None:
        label += f"summary\n{how}{'+' if indulge else ''}\n"
        if by_entropy:
            label += "Opacity: inverse entropy"
            max_coeff, _, opacity = most_resonant(mag_phase_mx[..., 0])
        else:
            label += "Opacity: magnitude"
            max_coeff, opacity, _ = most_resonant(mag_phase_mx[..., 0])
        colors = most_resonant2color(max_coeff, opacity)
    else:
        label += f"c{coeff}\n{how}{'+' if indulge else ''}"
        colors = circular_hue(mag_phase_mx[..., coeff - 1, :], output_rgba=True, ignore_phase=grey)
    colors = long2utm(colors)
    if colors.shape[-1] == 1:
        colors = colors[...,0]
    ws = Wavescape(colors, width=2286)
    ws.draw(label=label, aw_per_tick=10, tick_factor=10, label_size=20, indicator_size=1.0, tight_layout=False)
    plt.savefig(path)
    plt.close()



def store_mag_phase_mxs(debussy_repo, data_folder, norm_params, overwrite=False, cores=0, sort=False):
    print("Creating DFT matrices...", end=' ')
    dfts = get_dfts(debussy_repo, long=True)
    print('DONE')
    fpath2params = {os.path.join(data_folder, make_filename(k, how, indulge, ext='.npy.gz')): (k, how, indulge)
                    for k, (how, indulge) in product(dfts.keys(), norm_params)}

    if overwrite:
        pieces = len(dfts)
    else:
        print("Checking for existing files to be skipped...", end=' ')
        fpath2params = {fpath: params
                          for fpath, params in fpath2params.items()
                          if not os.path.isfile(fpath)}
        used_keys = set(params[0] for params in fpath2params.values())
        pieces = len(used_keys)
        delete_keys = [k for k in dfts.keys() if k not in used_keys]
        for k in delete_keys:
            del(dfts[k])
        print('DONE')
    n_runs = len(fpath2params)
    if n_runs == 0:
        print("No new magnitude-phase matrices to be computed.")
        return
    params = [(path, dfts[key], how, indulge)
              for path, (key, how, indulge) in fpath2params.items()]
    if sort:
        params = sorted(params, key=lambda t: t[1].shape[0])
    print(f"Computing {n_runs} magnitude-phase matrices for {pieces} pieces {core_msg(cores)}...")
    _ = do_it(compute_mag_phase_mx, params, n=n_runs, cores=cores)

def compute_mag_phase_mx(file_path, dft, how, indulge_prototypes):
    normalized = normalize_dft(dft, how=how, indulge_prototypes=indulge_prototypes)
    with gzip.GzipFile(file_path, "w") as zip_file:
        np.save(file=zip_file, arr=normalized)

def do_it(func, params, n=None, cores=0):
    if n is None:
        n = len(list(params))
    if cores == 0:
        return [func(*p) for p in tqdm(params, total=n)]
    pool = mp.Pool(cores)
    result = pool.starmap(func, tqdm(params, total=n))
    pool.close()
    pool.join()
    return result


def store_correlations(debussy_repo, data_folder, overwrite=False, cores=0, sort=False):
    print("Computing pitch-class-vector triangles...", end=' ')
    pcms = get_pcms(debussy_repo, long=True)
    print('DONE')
    pcms = {os.path.join(data_folder, fname + '-correlations.npy.gz'): pcm
              for fname, pcm in pcms.items()}
    if not overwrite:
        pcms = {path: pcm for path, pcm in pcms.items() if not os.path.isfile(path)}
    n = len(pcms)
    if n == 0:
        print("No new correlation matrices to be computed.")
        return
    params = list(pcms.items())
    if sort:
        params = sorted(params, key=lambda t: t[1].shape[0])
    print(f"Computing correlation matrices for {n} pieces {core_msg(cores)}...")
    _ = do_it(compute_correlations, params, n=n, cores=cores)


def compute_correlations(file_path, pcm):
    tritones = pitch_class_matrix_to_tritone(pcm)
    maj = max_pearsonr_by_rotation(pcm, 'mozart_major')
    min = max_pearsonr_by_rotation(pcm, 'mozart_minor')
    stacked = np.column_stack([maj, min, tritones])
    with gzip.GzipFile(file_path, "w") as zip_file:
        np.save(file=zip_file, arr=stacked)


def main(args):
    store_mag_phase_mxs(args.repo,
                        data_folder=args.data,
                        norm_params=args.normalization,
                        overwrite=args.magphase,
                        cores=args.cores,
                        sort=args.sort,
                        )
    store_correlations(args.repo,
                       data_folder=args.data,
                       overwrite=args.correlations,
                       cores=args.cores,
                       sort=args.sort,
                       )
    if args.wavescapes is not None:
        store_wavescapes(
            args.wavescapes,
            data_folder=args.data,
            norm_params=args.normalization,
            coeffs=args.coeffs,
            overwrite_standard=args.standard,
            overwrite_grey=args.grey,
            overwrite_summary=args.summary,
            cores=args.cores,
            sort=args.sort
        )






if __name__ == "__main__":
    NORM_METHODS = ['0c', 'post_norm', 'max_weighted', 'max']
    n_meth = 2 * len(NORM_METHODS)
    int2norm = dict(enumerate(NORM_METHODS + [norm + '+indulge' for norm in NORM_METHODS]))
    position2params = [(how, indulge) for indulge, how in product((False, True), NORM_METHODS)]
    parser = argparse.ArgumentParser(
        description="Create Debussy data."
    )
    parser.add_argument(
        "data",
        metavar="DATA_DIR",
        type=check_and_create,
        help="Directory where the NumPy arrays will be stored.",
    )
    parser.add_argument(
        "-w",
        "--wavescapes",
        metavar="WAVESCAPE_DIR",
        type=check_and_create,
        help="If you don't pass this argument, no wavescapes will be created."
    )
    parser.add_argument(
        "-n",
        "--normalization",
        nargs="+",
        metavar="METHOD",
        type=int,
        help=f"By default, all {n_meth} normalization methods are being applied. Pass one or "
             f"several numbers to use only some of them: {int2norm}."
    )
    parser.add_argument(
        "--coeffs",
        nargs="+",
        type=int,
        help="By default, wavescapes for all coefficients are created (if -w is set). Pass one "
             "or several numbers within [1, 6] to create only these."
    )
    parser.add_argument(
        "-c",
        "--cores",
        default=0,
        type=int,
        metavar="N",
        help="Defaults to 0, meaning that all available CPU cores are used in parallel to speed up "
             "the computation. Pass the desired number of cores or a negative number to deactivate."
    )
    parser.add_argument(
        "-s",
        "--sort",
        action='store_true',
        help="This flag influences the processing order by sorting the data matrices from shortest "
             "to longest"
    )
    parser.add_argument(
        "-r",
        "--repo",
        metavar="DIR",
        type=check_dir,
        default=os.getcwd(),
        help="Local clone of the debussy repository. Defaults to current working directory.",
    )

    overwriting_group = parser.add_argument_group(title='What kind of existing data to overwrite?')
    overwriting_group.add_argument(
        "--all",
        action='store_true',
        help="Set this flag to create all data from scratch. This amounts to all flags below."
    )
    overwriting_group.add_argument(
        "--magphase",
        action="store_true",
        help="Set this flag to re-compute normalized magnitude-phase matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--correlations",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--matrices",
        action="store_true",
        help="Short for --magphase --correlations"
    )
    overwriting_group.add_argument(
        "--standard",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--grey",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--summary",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--ws",
        action="store_true",
        help="Short for --standard --grey --summary. In other words, re-create all wavescapes but "
             "not the matrices."
    )

    args = parser.parse_args()
    # normalization param
    if args.normalization is None:
        args.normalization = position2params
    else:
        params = []
        for i in args.normalization:
            assert 0 <= i < n_meth, f"Arguments for -n need to be between 0 and {n_meth}, not {i}."
            params.append(position2params[i])
        args.normalization = params
    # multiprocessing param
    available_cpus = mp.cpu_count()
    if args.cores  == 0:
        args.cores = available_cpus
    elif args.cores > available_cpus:
        print(f"{args.cores} CPUs not available, setting the number down to {available_cpus}.")
        args.cores = available_cpus
    elif args.cores < 0:
        args.cores = 0 # deactivates multiprocessing
    # coefficients param
    if args.coeffs is not None:
        for i in args.coeffs:
            assert 0 < i < 7, f"Arguments for --coeff need to be within [1,6], not {i}"
    # overwriting params
    if args.all:
        args.matrices = True
        args.ws = True
    if args.matrices:
        args.magphase = True
        args.correlations = True
    if args.ws:
        args.standard = True
        args.grey = True
        args.summary = True
    main(args)


