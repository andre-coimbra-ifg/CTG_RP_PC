#!/usr/bin/env python
# coding: utf-8

import os
import json
from pprint import pprint

import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

import wfdb
from ctg_utils import get_all_recno, parse_meta_comments
from basic_denoise import (
    get_valid_segments,
    get_segment_concatenation,
    get_segment_removing_zeros,
)
from libGAF import create_gaf

POLICIES = ["best_quality", "early_valid", "late_valid"]


def generate_gaf_images(
    recordings_dir,
    n_dec=4,
    clip_stage_II=False,
    max_seg_min=10,
    policy="early_valid",
    images_dir="",
    images_index_file="gaf_images_index.json",
    show_signal=False,
    show_image=False,
    verbose=False,
    cmap=None,
    limit=-1,
):

    assert policy in POLICIES

    if images_dir and not os.path.exists(images_dir):
        os.mkdir(images_dir)

    max_seg = int(max_seg_min * 60 * 4)  # convert to samples

    results = {}
    for recno in sorted(get_all_recno(recordings_dir)):
        if limit == 0:
            break
        limit -= 1

        recno_full = os.path.join(recordings_dir, recno)
        all_sig, meta = wfdb.io.rdsamp(recno_full)
        meta = parse_meta_comments(meta["comments"])
        if verbose:
            print(
                "\nRecord: {}  Samples: {}   Duration: {:0.1f} min   Stage.II: {} min".format(
                    recno,
                    all_sig.shape[0],
                    all_sig.shape[0] / 4 / 60,
                    meta["Delivery"]["II.stage"],
                )
            )

        sig_hr = all_sig[:, 0]
        if clip_stage_II and meta["Delivery"]["II.stage"] != -1:
            idx = int(meta["Delivery"]["II.stage"] * 60 * 4)
            sig_hr = sig_hr[:-idx]
        ts = np.arange(len(sig_hr)) / 4.0

        if show_signal:
            plt.figure(figsize=(12, 2))
            plt.title("{}: Signal".format(recno))
            plt.plot(ts / 60, sig_hr)
            plt.xlim(ts[0], ts[-1] / 60)
            plt.ylim(50, 200)
            plt.show()

        # select segment with lowest error rate
        selected_segments = get_valid_segments(sig_hr, ts, recno, verbose=False)

        if len(selected_segments) == 0:
            continue

        if policy == "best_quality":
            selected_segments = sorted(selected_segments, key=lambda x: -x["pct_valid"])
        elif policy == "early_valid":
            selected_segments = sorted(selected_segments, key=lambda x: x["seg_start"])
        elif policy == "late_valid":
            selected_segments = sorted(selected_segments, key=lambda x: -x["seg_end"])

        seg = selected_segments[0]
        seg_start = seg["seg_start"]
        seg_end = seg["seg_end"]
        seg_hr = seg["seg_hr"]
        seg_tm = seg["seg_ts"] / 60
        orig_seg_hr = seg["orig_seg_hr"]
        mask = seg["mask"]
        pct_valid = seg["pct_valid"]

        if show_signal:
            plt.figure(figsize=(12, 2))
            plt.title("{}: Segment  {}-{}".format(recno, seg_start, seg_end))
            plt.plot(seg_tm, seg_hr)
            plt.plot(seg_tm, orig_seg_hr, alpha=0.25)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(50, 200)
            plt.show()

            plt.figure(figsize=(12, 2))
            plt.title("{}: Invalid".format(recno))
            plt.plot(seg_tm, ~mask)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(-0.1, 1.1)
            plt.show()

            print("Valid: {:0.1f}%".format(100 * pct_valid))

        if policy == "late_valid":
            selected_hr = seg_hr[-max_seg:]
        else:
            selected_hr = seg_hr[:max_seg]

        if n_dec > 1:
            selected_hr = scipy.signal.decimate(selected_hr, n_dec)

        image_names = []
        fname = create_gaf(
            selected_hr,
            base_name=recno,
            show_image=show_image,
            images_dir=images_dir,
            cmap=cmap,
            use_clip=clip_stage_II,
        )
        image_names.append(fname)

        results[recno] = {"names": image_names, "outcome": meta["Outcome"]}

    if verbose:
        pprint(results)

    with open(os.path.join(images_dir, images_index_file), "w") as outfile:
        json.dump(results, outfile)


def generate_gaf_images_segment(
    recordings_dir,
    n_dec=4,
    clip_stage_II=False,
    max_seg_min=10,
    policy="early_valid",
    images_dir="",
    images_index_file="gaf_images_index.json",
    show_signal=False,
    show_image=False,
    verbose=False,
    cmap=None,
    limit=-1,
):

    assert policy in POLICIES

    if images_dir and not os.path.exists(images_dir):
        os.mkdir(images_dir)

    results = {}
    for recno in sorted(get_all_recno(recordings_dir)):
        if limit == 0:
            break
        limit -= 1

        recno_full = os.path.join(recordings_dir, recno)
        all_sig, meta = wfdb.io.rdsamp(recno_full)
        meta = parse_meta_comments(meta["comments"])
        if verbose:
            print(
                "\nRecord: {}  Samples: {}   Duration: {:0.1f} min   Stage.II: {} min".format(
                    recno,
                    all_sig.shape[0],
                    all_sig.shape[0] / 4 / 60,
                    meta["Delivery"]["II.stage"],
                )
            )

        sig_hr = all_sig[:, 0]
        if clip_stage_II and meta["Delivery"]["II.stage"] != -1:
            idx = int(meta["Delivery"]["II.stage"] * 60 * 4)
            sig_hr = sig_hr[:-idx]
        ts = np.arange(len(sig_hr)) / 4.0

        # select concatenation of segments as close to the chosen length
        segment = get_segment_concatenation(
            sig_hr, ts, recno, max_seg_min=max_seg_min, policy=policy
        )

        if len(segment) == 0:
            continue

        if show_signal:
            seg_start = segment["seg_start"]
            seg_end = segment["seg_end"]
            seg_hr = segment["seg_hr"]
            seg_tm = segment["seg_ts"] / 60
            orig_seg_hr = segment["orig_seg_hr"]

            seg_tm = np.arange(len(seg_hr)) / 4.0 / 60

            plt.figure(figsize=(12, 3.8))
            plt.title(
                "{}: Final Segment  {}-{} (considered length = {:0.1f} min)".format(
                    recno, seg_start, seg_end, len(seg_hr) / 60 / 4
                )
            )
            plt.plot(seg_tm, seg_hr)
            plt.plot(seg_tm, orig_seg_hr, alpha=0.25)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(50, 200)
            plt.xlabel("Time [seconds]")
            plt.ylabel("FHR [bpm]")
            plt.show()

            print("Segment length: {:0.1f} min".format(len(seg_hr) / 60 / 4))

        selected_hr = segment["seg_hr"]

        # Reduce signal sampling rate (compact)
        if n_dec > 1:
            selected_hr = scipy.signal.decimate(selected_hr, n_dec)

        selected_hr = np.round(selected_hr).astype(int)

        # Keep list to maintain structure used in other methods
        image_names = []
        fname = create_gaf(
            selected_hr,
            base_name=recno,
            show_image=show_image,
            images_dir=images_dir,
            cmap=cmap,
        )  # , use_clip=clip_stage_II)

        image_names.append(fname)

        results[recno] = {"names": image_names, "outcome": meta["Outcome"]}

    if verbose:
        pprint(results)

    with open(os.path.join(images_dir, images_index_file), "w") as outfile:
        json.dump(results, outfile)

    return len(results)


def generate_gaf_images_simple(
    recordings_dir,
    n_dec=4,
    clip_stage_II=False,
    max_seg_min=10,
    policy="early_valid",
    rp_params=[{}],
    images_dir="",
    images_index_file="gaf_images_index.json",
    show_signal=False,
    show_image=False,
    verbose=False,
    cmap=None,
    limit=-1,
):

    assert policy in POLICIES

    if images_dir and not os.path.exists(images_dir):
        os.mkdir(images_dir)

    results = {}
    for recno in sorted(get_all_recno(recordings_dir)):
        if limit == 0:
            break
        limit -= 1

        recno_full = os.path.join(recordings_dir, recno)
        all_sig, meta = wfdb.io.rdsamp(recno_full)
        meta = parse_meta_comments(meta["comments"])
        if verbose:
            print(
                "\nRecord: {}  Samples: {}   Duration: {:0.1f} min   Stage.II: {} min".format(
                    recno,
                    all_sig.shape[0],
                    all_sig.shape[0] / 4 / 60,
                    meta["Delivery"]["II.stage"],
                )
            )

        sig_hr = all_sig[:, 0]
        if clip_stage_II and meta["Delivery"]["II.stage"] != -1:
            idx = int(meta["Delivery"]["II.stage"] * 60 * 4)
            sig_hr = sig_hr[:-idx]
        ts = np.arange(len(sig_hr)) / 4.0

        # select concatenation of segments as close to the chosen length
        segment = get_segment_removing_zeros(
            sig_hr, ts, recno, max_seg_min=max_seg_min, policy=policy
        )

        if len(segment) == 0:
            continue

        selected_hr = segment["seg_hr"]

        if show_signal:
            seg_tm = np.arange(len(selected_hr)) / 4.0 / 60

            plt.figure(figsize=(12, 3.8))
            plt.title(
                "{}: Final Segment (considered length = {:0.1f} min)".format(
                    recno, len(selected_hr) / 60 / 4
                )
            )
            plt.plot(seg_tm, selected_hr)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(50, 200)
            plt.xlabel("Time [seconds]")
            plt.ylabel("FHR [bpm]")
            plt.show()

        # Reduce signal sampling rate (compact)
        if n_dec > 1:
            selected_hr = scipy.signal.decimate(selected_hr, n_dec)

        selected_hr = np.round(selected_hr).astype(int)

        image_names = []
        fname = create_gaf(
            selected_hr,
            base_name=recno,
            show_image=show_image,
            images_dir=images_dir,
            cmap=cmap,
            use_clip=clip_stage_II,
        )
        image_names.append(fname)

        results[recno] = {"names": image_names, "outcome": meta["Outcome"]}

    with open(os.path.join(images_dir, images_index_file), "w") as outfile:
        json.dump(results, outfile)
