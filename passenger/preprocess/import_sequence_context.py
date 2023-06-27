import pandas as pd
import numpy as np


def get(chrom, pos, tpe="", window=30, norm=True, path=""):
    rng = np.arange(pos - window, pos + window + 1)
    ALT = pd.read_csv(path + tpe + "/ALT_" + chrom + "_" + str(pos) + ".csv", header=None)
    REF = pd.read_csv(path + tpe + "/REF_" + chrom + "_" + str(pos) + ".csv", header=None)

    uniq, uniq_idx, counts = np.unique(ALT[1], axis=0, return_index=True, return_counts=True)
    to_drop = []
    for i in uniq[counts > 1]:
        whr = np.where(ALT[1] == i)[0]
        to_drop.extend(np.delete(whr, 0))

    ALT = ALT.drop(ALT.index[to_drop])
    ALT.index = ALT[1]
    missing = np.where([i not in ALT.index for i in rng])[0]
    to_add = pd.DataFrame([[chrom, rng[i], 0, 0, 0, 0, np.nan] for i in missing])
    to_add.index = rng[missing]
    ALT = pd.concat((ALT, to_add))

    uniq, uniq_idx, counts = np.unique(REF[1], axis=0, return_index=True, return_counts=True)
    to_drop = []
    for i in uniq[counts > 1]:
        whr = np.where(REF[1] == i)[0]
        to_drop.extend(np.delete(whr, 0))

    REF = REF.drop(REF.index[to_drop])
    REF.index = REF[1]
    missing = np.where([i not in REF.index for i in rng])[0]
    to_add = pd.DataFrame([[chrom, rng[i], 0, 0, 0, 0, np.nan] for i in missing])
    to_add.index = rng[missing]
    REF = pd.concat((REF, to_add))

    REF = REF.loc[rng]
    ALT = ALT.loc[rng]

    REF = REF.drop(columns=[0, 1, 6])
    ALT = ALT.drop(columns=[0, 1, 6])

    if norm:
        REF /= np.sum(REF.loc[pos])
        ALT /= np.sum(ALT.loc[pos])

    return REF, ALT


def get_variant_as_single_row(chrom, pos, tpe="", window=30, path=""):
    REF, ALT = get(chrom, pos, tpe, window=30, path=path)

    REF.columns = ["ref_A", "ref_T", "ref_G", "ref_C"]
    ALT.columns = ["alt_A", "alt_T", "alt_G", "alt_C"]

    REF.index = np.arange(-window, window + 1, 1).astype(str)
    ALT.index = np.arange(-window, window + 1, 1).astype(str)

    single_row = pd.concat((ALT, REF), axis=1).unstack().to_frame().T
    single_row.columns = single_row.columns.map('_'.join)

    return single_row


def get_variant_as_matrix(chrom, pos, tpe="", window=30, path=""):
    REF, ALT = get(chrom, pos, tpe, window=window, path=path)
    return np.array(pd.concat((REF, ALT), axis=1)[[2, 3, 4, 5]])


"""
import matplotlib.pyplot as plt

def plot_variant(chrom, pos, tpe="", window=30, path=""):
    REF, ALT = get(chrom, pos, tpe, window, False, path=path)

    matched = (np.argmax(np.array(REF[[2, 3, 4, 5]]), axis=1) != np.argmax(np.array(ALT[[2, 3, 4, 5]]), axis=1)).astype(
        int)

    fig, axs = plt.subplots(3, 1, figsize=(20, 4))

    axs[0].bar(REF.index, REF[2], alpha=.5)
    axs[0].bar(REF.index, REF[3], bottom=REF[2], alpha=.5)
    axs[0].bar(REF.index, REF[4], bottom=REF[2] + REF[3], alpha=.5)
    axs[0].bar(REF.index, REF[5], bottom=REF[2] + REF[3] + REF[4], alpha=.5, color="goldenrod")
    # axs[0].set_yticks([]),
    axs[0].set_xticks([])
    axs[0].set_ylabel("REF", size=20)

    axs[1].bar(REF.index, ALT[2], alpha=.5)
    axs[1].bar(REF.index, ALT[3], bottom=ALT[2], alpha=.5)
    axs[1].bar(REF.index, ALT[4], bottom=ALT[2] + ALT[3], alpha=.5)
    axs[1].bar(REF.index, ALT[5], bottom=ALT[2] + ALT[3] + ALT[4], alpha=.5, color="goldenrod")
    # axs[1].set_yticks([]),
    axs[1].set_xticks([])
    axs[1].set_ylabel("ALT", size=20)

    axs[2].bar(REF.index, matched, alpha=.5, color="firebrick")
    axs[2].set_yticks([]), axs[2].set_xticks([])

    plt.tight_layout()
    plt.show()

"""
