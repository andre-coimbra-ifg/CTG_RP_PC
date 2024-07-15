#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from pprint import pprint

POLICIES = ['best_quality', 'early_valid', 'late_valid']


def find_valid_start(sig, n_stable=1, min_delta=25):
    for i in range(len(sig)-n_stable):
        max_value = np.max(sig[i:i+n_stable])
        min_value = np.min(sig[i:i+n_stable])

        if max_value == 0:
            continue
        if max_value - min_value > min_delta:
            continue
        return i

    return None


def find_gaps(sig):
    missing = sig == 0
    i_start = 0
    gaps = []
    while i_start < len(missing):
        i_start = np.argmax(missing[i_start:]) + i_start  # start of gap
        if not missing[i_start]:
            break

        i_end = np.argmin(missing[i_start:]) + i_start   # end of gap
        if i_end == i_start:
            i_end = len(missing)   # reached end

        n_gap = i_end - i_start
        gaps.append([n_gap, i_start, i_end])
        i_start = i_end

    return gaps


# Remove segments that last less than 3 seconds and last less than the adjacent gaps
def trim_short_segments(sig, verbose=False, min_seg=3*4):
    gaps = find_gaps(sig)
    for i in range(1, len(gaps)):
        n_seg = gaps[i][1] - gaps[i-1][2]
        if n_seg <= min_seg and n_seg < min(gaps[i-1][0], gaps[i][0]):
            sig[gaps[i-1][2]:gaps[i][1]] = 0
            if verbose:
                print('n_seg', n_seg,  gaps[i-1][0], gaps[i][0])
    return sig


# Remove missing values segments that last longer than 15s
# If min_segment_width not set, it considers only segments greater than 8 minutes
def find_valid_segments(sig, min_segment_width=8*60*4,
                        max_allowed_gap=15*4, verbose=False):

    gaps = find_gaps(sig)
    gaps = [g for g in gaps if g[0] > max_allowed_gap]
    if verbose:
        for g in gaps:
            print('gap @ {:0.2f} min for {:0.2f} sec  index: {} '.format(
                g[1]/4/60, g[0]/4, g[1:]))

    n_sig = len(sig)
    valid_segments = []
    seg_start = 0

    for _, gap_start, gap_end in gaps:
        seg_end = gap_start               # gap ends valid segment
        n_seg = seg_end - seg_start
        if n_seg >= min_segment_width:    # ignore short segments
            valid_segments.append([seg_start, seg_end])
        seg_start = gap_end

    if seg_start < n_sig:     # special case for final segment
        seg_end = n_sig
        n_seg = seg_end - seg_start
        if n_seg >= min_segment_width:
            valid_segments.append([seg_start, seg_end])

    if verbose:
        print('valid_segments')
        pprint(valid_segments)
    return valid_segments


def filter_missing_values(sig):
    valid = sig > 0
    n_valid = np.sum(valid)
    n_missing = len(valid) - n_valid

    if n_missing > 0 and n_valid > 0:
        x = np.arange(len(valid))
        sig = np.interp(x, x[valid], sig[valid])

    return sig, valid

# Remove spikes (signal value greater than 200bpm e less than 50 bpm)
# and fill them with Hermite Spline Interpolation


def filter_extreme_values(seg_hr, valid, min_hr=50, max_hr=200):
    seg_hr[seg_hr < min_hr] = 0
    seg_hr[seg_hr > max_hr] = 0

    change_mask = seg_hr > 0
    n_valid = np.sum(change_mask)
    n_missing = len(change_mask) - n_valid

    if n_missing > 0 and n_valid > 0:
        x = np.arange(len(change_mask))
        pchip = PchipInterpolator(x[change_mask], seg_hr[change_mask])
        seg_hr = pchip(x)

        valid[~change_mask] = False

    return seg_hr, valid


# The difference between two adjacent points can't be greater than 25 bpm
def filter_large_changes(sig, valid, tm, max_change=25, verbose=False):
    sig_d = np.abs(np.diff(sig))
    change_mask = sig_d > max_change

    change_mask = np.logical_and(
        change_mask, np.logical_and(valid[1:], valid[:-1]))

    change_mask = np.logical_or(
        np.pad(change_mask, (1, 0), 'edge'),
        np.logical_or(np.pad(change_mask, (2, 0), 'edge')[:-1],
                      np.pad(change_mask, (3, 0), 'edge')[:-2]))

    if np.sum(change_mask) > 0:
        x = np.arange(len(change_mask))
        sig = np.interp(x, x[~change_mask], sig[~change_mask])
        valid[change_mask] = False

    return sig, valid


def get_valid_segments(orig_hr, ts, recno, max_change=25, verbose=False, verbose_details=False):
    """Returns valid segments ordered by error rate"""

    if verbose:
        plt.figure(figsize=(12, 2))
        plt.title('{}: Full Recording (orig)'.format(recno))
        plt.plot(ts / 60, orig_hr)
        plt.ylim(0, 240)
        plt.xlim(ts[0]/60, ts[-1]/60)
        plt.show()

    i_start = find_valid_start(orig_hr)

    if i_start is None:
        return []

    orig_hr = orig_hr[i_start:]
    sig_hr = np.copy(orig_hr)
    ts = ts[i_start:]
    tm = ts / 60

    # sig_hr = trim_short_segments(sig_hr, verbose=verbose_details)

    # 1. Remove gaps greater than 15 seconds
    valid_segments = find_valid_segments(
        sig_hr, min_segment_width=2*60*4, verbose=verbose_details)
    #  min_segment_width=0,

    selected_segments = []
    for seg_start, seg_end in valid_segments:
        # get segment
        seg_hr = sig_hr[seg_start:seg_end]
        seg_ts = ts[seg_start:seg_end]
        seg_tm = tm[seg_start:seg_end]

        # adjust for stability at start of recording
        new_start = find_valid_start(seg_hr)
        if new_start is None:
            print('unable to find stable region')
            continue
        elif new_start != seg_start:
            seg_start = seg_start + new_start
            seg_hr = sig_hr[seg_start:seg_end]
            seg_ts = ts[seg_start:seg_end]
            seg_tm = tm[seg_start:seg_end]

        # 2. Linear interpolation on small gaps
        seg_hr, mask = filter_missing_values(seg_hr)

        # 3. Linear interpolation to stabilize signal (diff < 25bpm)
        seg_hr, mask = filter_large_changes(
            seg_hr, mask, seg_tm, max_change=max_change, verbose=verbose_details)

        # 4. Hermite Spline Interpolation on spikes (50 < fhr < 200)
        seg_hr, mask = filter_extreme_values(seg_hr, mask)

        selected_segments.append(
            {'seg_start': seg_start,
             'seg_end': seg_end,
             'seg_hr': seg_hr,
             'seg_ts': seg_ts,
             'orig_seg_hr': orig_hr[seg_start:seg_end],
             'mask': mask,
             'pct_valid': np.mean(mask)
             })

    # order valid segments by error rate
    selected_segments = sorted(
        selected_segments, key=lambda x: -x['pct_valid'])

    if verbose:
        for seg in selected_segments:
            seg_start = seg['seg_start']
            seg_end = seg['seg_end']
            seg_hr = seg['seg_hr']
            seg_tm = seg['seg_ts'] / 60
            orig_seg_hr = seg['orig_seg_hr']
            mask = seg['mask']
            pct_valid = seg['pct_valid']

            plt.figure(figsize=(12, 2))
            plt.title('{}: Final Signal  {}-{}'.format(recno, seg_start, seg_end))
            plt.plot(seg_tm, seg_hr)
            plt.plot(seg_tm, orig_seg_hr, alpha=0.25)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(50, 200)
            plt.show()

            plt.figure(figsize=(12, 2))
            plt.title('{}: Invalid'.format(recno))
            plt.plot(seg_tm, ~mask)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(-0.1, 1.1)
            plt.show()

            if verbose_details:
                plt.figure(figsize=(12, 2))
                plt.title(
                    '{}: Final Signal diff  {}-{}'.format(recno, seg_start, seg_end))
                plt.plot(seg_tm[:-1], np.diff(seg_hr))
                plt.plot([seg_tm[0], seg_tm[-1]], [0, 0], 'r--')
                plt.plot([seg_tm[0], seg_tm[-1]],
                         [max_change, max_change], 'k--')
                plt.plot([seg_tm[0], seg_tm[-1]],
                         [-max_change, -max_change], 'k--')
                plt.xlim(seg_tm[0], seg_tm[-1])
                plt.show()

            print('Valid: {:0.1f}%'.format(100 * pct_valid))

    return selected_segments


def save_xlsx(result, name):

    # selected_columns = ['seg_hr', 'seg_ts', 'orig_seg_hr', 'mask']
    selected_columns = ['seg_hr']

    new_data = {col: result[col] for col in selected_columns}

    df = pd.DataFrame(new_data)
    df = df.round(2)

    file_path = 'result_'+name+'.xlsx'

    df.to_excel(file_path, index=False)


def get_segment_concatenation(orig_hr, ts, recno, max_seg_min=3, policy='early_valid'):
    """Returns a concatenation of valid segments as close as possible to the desired length in minutes"""

    # get segments with lowest error rate
    selected_segments = get_valid_segments(orig_hr, ts, recno)

    if len(selected_segments) == 0:
        return {}

    max_seg = max_seg_min*60*4  # convert minutes to samples

    if policy == 'early_valid':
        selected_segments = sorted(
            selected_segments, key=lambda x: x['seg_start'])
    elif policy == 'late_valid':
        selected_segments = sorted(
            selected_segments, key=lambda x: -x['seg_end'])

    seg_columns = ['seg_hr', 'seg_ts', 'orig_seg_hr', 'mask']

    result = {'seg_start': [], 'seg_end': [], 'seg_hr': [],
              'seg_ts': [], 'orig_seg_hr': [], 'mask': np.bool_([]), 'pct_valid': None}

    if policy == 'best_quality' or policy == 'early_valid':
        for seg in selected_segments:
            if len(result['seg_hr']) >= max_seg:
                break
            for col in seg_columns:
                result[col] = np.concatenate(
                    [result[col], seg[col]])

        # save_xlsx(result, 'initial')
        for col in seg_columns:
            result[col] = result[col][:max_seg]

    elif policy == 'late_valid':
        for seg in selected_segments:
            if len(result['seg_hr']) >= max_seg:
                break

            for col in seg_columns:
                result[col] = np.concatenate(
                    [seg[col], result[col]])

        # save_xlsx(result, 'initial')
        for col in seg_columns:
            result[col] = result[col][-max_seg:]

    result['seg_start'] = np.int64((result['seg_ts'][0]*4))
    result['seg_end'] = np.int64((result['seg_ts'][-1]*4)+1)
    result['pct_valid'] = np.mean(result['mask'])
    # save_xlsx(result, 'final')

    return result


def get_segment_removing_zeros(orig_hr, ts, recno, max_seg_min=3, policy='early_valid'):
    """Returns a concatenation of valid segments as close as possible to the desired length in minutes"""

    result = {'seg_start': [], 'seg_end': [], 'seg_hr': [],
              'seg_ts': [], 'orig_seg_hr': [], 'mask': np.bool_([]), 'pct_valid': None}

    seg_hr = orig_hr[orig_hr != 0]
    max_seg = max_seg_min*60*4  # convert minutes to samples

    if policy == 'late_valid':
        result['seg_hr'] = seg_hr[-max_seg:]
    else:
        result['seg_hr'] = seg_hr[:max_seg]

    # save_xlsx(result, 'final')

    return result
