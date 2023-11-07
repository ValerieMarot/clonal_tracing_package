import pandas as pd
import numpy as np
from passenger.preprocess.filter import get_pars_from_line, filter_vars


def get_raw_data(raw_prefix, filter_germline=None, filter_ref_alt_cells=None):
    ALT = pd.read_csv(raw_prefix + "ALT.csv", index_col=0)
    REF = pd.read_csv(raw_prefix + "REF.csv", index_col=0)
    meta = pd.read_csv(raw_prefix + "meta.csv", index_col=0)

    if filter_germline is not None and filter_ref_alt_cells is not None:
        REF, ALT, meta = filter_vars(REF, ALT, meta, filter_germline=filter_germline,
                                     filter_ref_alt_cells=filter_ref_alt_cells)

    return REF, ALT, meta


def get_run_data(file_prefix):
    C = np.loadtxt(file_prefix + "_C.csv", delimiter=",")
    C_std = np.loadtxt(file_prefix + "_C_std.csv", delimiter=",")
    V = np.loadtxt(file_prefix + "_V.csv", delimiter=",")
    V_std = np.loadtxt(file_prefix + "_V_std.csv", delimiter=",")
    meta = pd.read_csv(file_prefix + "_meta.csv", delimiter=",", index_col=0)
    return C, C_std, V, V_std, meta


def get_best_run(run_prefix, raw_prefix, parfile, k=2):
    best_score, best_line = 0, -1
    C, C_std, V, V_std, meta = None, None, None, None, None

    for i in np.arange(1, 7):
        filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, i)
        file_prefix = run_prefix + "_" + str(filter_germline) + "_" + str(filter_ref_alt_cells)
        C_, C_std_, V_, V_std_, meta_ = get_run_data(file_prefix)

        if V_.shape[0] > 100:
            new_score = np.nanmean(np.abs(C_[0] - C_[1]))
            if k == 3:
                alldists = np.array((C_[0] - C_[1], C_[2] - C_[1], C_[0] - C_[2]))
                new_score = np.nanmean(np.abs(alldists))
            print(i, "\t", np.round(new_score, 2))
            if best_score < new_score:
                C, C_std, V, V_std, meta = C_, C_std_, V_, V_std_, meta_
                best_score = new_score
                best_line = i

    readme = (best_line, best_score)

    filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, best_line)
    REF, ALT, _ = get_raw_data(raw_prefix, filter_germline, filter_ref_alt_cells)
    cell_assignments = pd.DataFrame(C.T, index=ALT.columns)

    return cell_assignments, C_std, V, V_std, REF, ALT, meta, readme


def get_best_run_multik(run_prefix, patient, raw_prefix, parfile):
    run_prefix_k2 = run_prefix + "k2_" + patient
    cell_assignments, C_std, V, V_std, REF, ALT, meta, readme = get_best_run(run_prefix_k2, raw_prefix, parfile, k=2)
    try:
        run_prefix_k3 = run_prefix + "k3_" + patient
        pars = get_best_run(run_prefix_k3, raw_prefix, parfile, k=3)
        k3 = pars[0]
        corr = np.array([np.corrcoef(k3[0], k3[1])[1, 0],
                         np.corrcoef(k3[0], k3[2])[1, 0],
                         np.corrcoef(k3[2], k3[1])[1, 0]])
        n_cells = np.unique(np.argmax(k3, axis=1), return_counts=True)[1]

        if np.all(corr < 0) & np.all(n_cells > .05 * np.sum(n_cells)):
            cell_assignments, C_std, V, V_std, REF, ALT, meta, readme = pars
    except FileNotFoundError:
        print("No run for k 3")
    to_return = {"cell_assignments": cell_assignments, "C_std": C_std, "V": V, "V_std": V_std, "REF": REF, "ALT": ALT,
                 "meta": meta, "readme": readme}
    return to_return
