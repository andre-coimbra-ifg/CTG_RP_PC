#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import random


def get_all_recno(dbdir):
    for f in os.listdir(dbdir):
        if os.path.isfile(os.path.join(dbdir, f)) and f.endswith(".hea"):
            yield f.split(".")[0]


def parse_meta_comments(comments, verbose=False):
    result = {}
    for c in comments:
        # print(c)
        if c.startswith("---"):
            continue
        if c.startswith("-- "):
            entry = {}
            k = c.split()[1]
            result[k] = entry
            continue
        c = c.strip()
        idx = c.rfind(" ")
        k = c[:idx].strip()
        v = c[idx + 1 :]

        try:
            v = int(v)
        except Exception:
            try:
                v = float(v)
            except Exception:
                pass
        entry[k] = v
        if verbose:
            print("  {}:{} ({})".format(k, v, type(v)))
    return result


def physionet_ctg_generate_mask(sig):
    mask = sig != 0
    all_idx = np.arange(len(sig))

    new_sig = np.interp(all_idx, all_idx[mask], sig[mask])
    return mask, new_sig


def extract_ph(file_path):
    """Extracts the pH value from a .hea file."""
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#pH"):
                try:
                    return float(line.split()[1])
                except (IndexError, ValueError):
                    return None
    return None


def balance_files(directory, threshold, verbose=False):
    """Balances the number of files with pH above and below the threshold and removes unused .hea and associated .dat files."""
    above = []
    below = []

    # Collect files and classify them
    for filename in os.listdir(directory):
        if filename.endswith(".hea"):
            file_path = os.path.join(directory, filename)
            ph_value = extract_ph(file_path)
            if ph_value is not None:
                if ph_value >= threshold:
                    above.append(file_path)
                else:
                    below.append(file_path)

    # Determine the number of files to keep
    min_count = min(len(above), len(below))

    # Shuffle lists before removing files
    random.shuffle(above)
    random.shuffle(below)

    def remove_files(file_path):
        """Removes the .hea file and its corresponding .dat file."""
        os.remove(file_path)
        dat_file = file_path.replace(".hea", ".dat")
        if os.path.exists(dat_file):
            os.remove(dat_file)

    while len(above) > min_count:
        remove_files(above.pop())
    while len(below) > min_count:
        remove_files(below.pop())

    # Remove remaining unused files
    remaining_files = set(os.listdir(directory))
    for filename in remaining_files:
        if (
            filename.endswith(".hea")
            and os.path.join(directory, filename) not in above + below
        ):
            remove_files(os.path.join(directory, filename))

    if verbose:
        print(f"Balancing completed. {min_count} files in each category.")
