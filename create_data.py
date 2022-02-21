import os, argparse
import gzip
from itertools import product
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from wavescapes import normalize_dft  # pip install https://github.com/DCMLab/wavescapes/archive/refs/heads/johannes.zip

from etl import get_dfts, resolve_dir, get_pcms
from utils import pitch_class_matrix_to_tritone, max_pearsonr_by_rotation


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

def make_filename(fname, how, indulge_prototypes, coeff=None, ext=None):
    result = fname
    if coeff is not None:
        result += f"-c{coeff}"
    result += f"-{how}"
    if indulge_prototypes:
        result += "+indulge"
    if ext is not None:
        result += ext
    return result

def make_filenames(fnames, norm_params, coeffs=None, ext=None):
    if coeffs is None:
        coeffs = [None]
    ext = [ext]
    return [make_filename(fname, how, indulge, coeff, e)
            for fname, (how, indulge), coeff, e
            in product(fnames, norm_params, coeffs, ext)]


def store_mag_phase_mxs(debussy_repo, data_path, norm_params, overwrite=False, cores=0, sort=False):
    print("Creating DFT matrices...", end=' ')
    dfts = get_dfts(debussy_repo, long=True)
    print('DONE')
    fpath2params = {os.path.join(data_path, make_filename(k, how, indulge, ext='.npy.gz')): (k, how, indulge)
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
    print(f"Computing {n_runs} magnitude-phase matrices for {pieces} pieces...")
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


def store_correlations(debussy_repo, data_path, overwrite=False, cores=0, sort=False):
    print("Computing pitch-class-vector triangles...", end=' ')
    pcms = get_pcms(debussy_repo, long=True)
    print('DONE')
    pcms = {os.path.join(data_path, fname + '-correlations.npy.gz'): pcm
              for fname, pcm in pcms.items()}
    if not overwrite:
        pcms = {path: pcm for path, pcm in pcms.items() if not os.path.isfile(path)}
    params = list(pcms.items())
    if sort:
        params = sorted(params, key=lambda t: t[1].shape[0])
    n = len(params)
    print(f"Computing correlation matrices for {n} pieces...")
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
                        data_path=args.output,
                        norm_params=args.normalization,
                        overwrite=args.magphase,
                        cores=args.cores,
                        sort=args.sort,
                        )
    store_correlations(args.repo,
                       data_path=args.output,
                       overwrite=args.correlations,
                       cores=args.cores,
                       sort=args.sort,
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
        "output",
        metavar="OUTPUT_DIR",
        type=check_and_create,
        help="Directory where the data will be stored.",
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
        "-c",
        "--cores",
        default="0",
        metavar="N",
        help="If you want speed by using several CPU cores in parallel, either pass the desired "
             "number of cores, or a negative number to use all cores. Defaults to 0."
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
             "already in the output directory."
    )
    overwriting_group.add_argument(
        "--correlations",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the output directory."
    )


    args = parser.parse_args()
    if args.normalization is None:
        args.normalization = position2params
    else:
        params = []
        for i in args.normalization:
            assert 0 <= i < n_meth, f"Arguments for -n need to be between 0 and {n_meth}, not {i}."
            params.append(position2params[i])
        args.normalization = params
    args.cores = int(args.cores)
    available_cpus = mp.cpu_count()
    if args.cores < 0:
        args.cores = available_cpus
    elif args.cores > available_cpus:
        print(f"{args.cores} CPUs not available, setting the number down to {available_cpus}.")
        args.cores = available_cpus
    if args.all:
        args.magphase = True
        args.correlations = True
    main(args)


