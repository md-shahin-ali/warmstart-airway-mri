# ===============================================================
#  config.py  --  the ONLY file you need to edit
#
#  Every run_*.py script and every analysis script reads its paths from
#  here, so the whole pipeline points at one data file.
# ===============================================================
import os
import glob

# ---------------------------------------------------------------
#  1. PASTE YOUR DATA PATH HERE
# ---------------------------------------------------------------
#  Full path to ONE subject's k-space .mat file (MATLAB v7.3), e.g.
#      '/home/you/data/subject01/subject01.mat'
#  See README.md -> "Data format" for the variables it must contain.
#
#  You can also leave this line alone and set an environment variable
#  instead (handy on a cluster):
#      export WARMSTART_DATA_FILE=/home/you/data/subject01/subject01.mat
DATA_FILE = os.environ.get('WARMSTART_DATA_FILE',
                           'PASTE/YOUR/DATA/PATH/HERE/subject.mat')

# ---------------------------------------------------------------
#  2. A short label for this subject (used in figure titles / file names)
# ---------------------------------------------------------------
SUBJECT = os.environ.get('WARMSTART_SUBJECT', 'subject01')

# ---------------------------------------------------------------
#  Everything below is derived - no need to edit.
# ---------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RECON_DIR = REPO_DIR          # all .py files live in the repository root

# All outputs (pickle cache, checkpoints, reconstructions, logs) are written
# next to the data file, in folders that start with the file's stem.
DATA_DIR = os.path.dirname(os.path.abspath(DATA_FILE))
DATA_STEM = os.path.splitext(os.path.basename(DATA_FILE))[0]
ANALYSIS_DIR = os.path.join(DATA_DIR, f'{DATA_STEM}_analysis')


def resolve_data_file(slice_for_cache=None):
    """Return the path to hand to the reconstruction code.

    Normally that is simply DATA_FILE. But after a first run, every slice is
    cached as <stem>_<zero-based slice>.pickle next to the .mat, and the
    loader reads the cache instead of the .mat. So if the .mat has since been
    archived or deleted, the pipeline can still run as long as the cache for
    the slice it needs is present. slice_for_cache is the 1-based slice whose
    cache to check for in that case.
    """
    if 'PASTE/YOUR/DATA/PATH' in DATA_FILE:
        raise SystemExit('\n  config.py: DATA_FILE is still the placeholder.\n'
                         '  Open config.py and paste the full path to your .mat file,\n'
                         '  or set the WARMSTART_DATA_FILE environment variable.\n')
    if os.path.isfile(DATA_FILE):
        return DATA_FILE

    if slice_for_cache is not None:
        cache = DATA_FILE[:-len('.mat')] + f'_{slice_for_cache - 1}.pickle'
        if os.path.isfile(cache):
            print(f'  {DATA_FILE} not found, but its slice cache exists:\n'
                  f'    {cache}\n  continuing with the cache.', flush=True)
            return DATA_FILE
    else:
        caches = glob.glob(DATA_FILE[:-len('.mat')] + '_*.pickle')
        if caches:
            print(f'  {DATA_FILE} not found, but {len(caches)} slice cache file(s) '
                  f'exist next to it - continuing with the cache.', flush=True)
            return DATA_FILE

    raise SystemExit(f'\n  config.py: data file not found:\n    {DATA_FILE}\n'
                     f'  Check the path in config.py.\n')
