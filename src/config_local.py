#
# Colab Notebook Configuration
#

GITHUB_PREFIX = (
    "https://raw.githubusercontent.com/andre-coimbra-ifg/CTG_RP_PC/master/src/"
)

GITHUB_DEFAULT_SRC_FILES = [
    "basic_denoise.py",
    "compute_metadata.py",
    "ctg_utils.py",
    "generate_recurrence_images.py",
    "generate_poincare_images.py",
    "generate_gramian_angular_field_images.py",
    "generate_markov_transition_field_images.py",
    "libPC.py",
    "libRP.py",
    "libGAF.py",
    "libMTF.py",
    "simple_test.py",
]

GITHUB_BASIC_SRC_FILES = [
    "basic_denoise.py",
    "compute_metadata.py",
    "ctg_utils.py",
    "simple_test.py",
]

GITHUB_RP_SRC_FILES = [
    "generate_recurrence_images.py",
    "libRP.py",
] + GITHUB_BASIC_SRC_FILES

GITHUB_PC_SRC_FILES = [
    "generate_poincare_images.py",
    "libPC.py",
] + GITHUB_BASIC_SRC_FILES

GITHUB_GAF_SRC_FILES = [
    "generate_gramian_angular_field_images.py",
    "libGAF.py",
] + GITHUB_BASIC_SRC_FILES

GITHUB_MTF_SRC_FILES = [
    "generate_markov_transition_field_images.py",
    "libMTF.py",
] + GITHUB_BASIC_SRC_FILES


RECORDINGS_DIR = "/content/ctu-uhb-ctgdb"
IMAGES_DIR = "/content/images"
THRESHOLD_PH = 7.15


def try_remove_python_file(fname):
    from pathlib import Path

    try:
        import_file = Path() / fname
        import_file.unlink()
    except:
        pass


def try_remove_python_file_old(fname):
    import os

    try:
        os.remove(fname)
    except:
        pass


def get_github_files(flist):
    import urllib.request

    for fname in flist:
        try:
            if "*" not in fname and "/" not in fname and "?" not in fname:
                print("Beginning file download of file", fname)
                print(GITHUB_PREFIX + fname)

                try_remove_python_file(fname)
                urllib.request.urlretrieve(GITHUB_PREFIX + fname, fname)
            else:
                print("Skipping file download of file", fname)
        except:
            print("Download failed for file", fname)
    print("Done")


def get_default_github_src_files():
    get_github_files(GITHUB_DEFAULT_SRC_FILES)


def get_rp_github_src_files():
    get_github_files(GITHUB_RP_SRC_FILES)


def get_pc_github_src_files():
    get_github_files(GITHUB_PC_SRC_FILES)


def get_gaf_github_src_files():
    get_github_files(GITHUB_GAF_SRC_FILES)


def get_mtf_github_src_files():
    get_github_files(GITHUB_MTF_SRC_FILES)
