import pandas as pd
import numpy as np
from passenger.preprocess.filter import get_pars_from_line, filter_vars
import os
from passenger.cluster.NMF import *
import matplotlib.pyplot as plt
from itertools import pairwise
#from kneed import KneeLocator


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def get_raw_data(raw_prefix, filter_germline=None, filter_ref_alt_cells=None):
    ALT = pd.read_csv(raw_prefix + "ALT.csv", index_col=0)
    REF = pd.read_csv(raw_prefix + "REF.csv", index_col=0)
    meta = pd.read_csv(raw_prefix + "meta.csv", index_col=0)

    if filter_germline is not None and filter_ref_alt_cells is not None:
        REF, ALT, meta = filter_vars(REF, ALT, meta, filter_germline=filter_germline,
                                     filter_ref_alt_cells=filter_ref_alt_cells)

    return REF, ALT, meta


def get_run_data(file_prefix):
    C = np.array(np.loadtxt(file_prefix + "_C.csv", delimiter=","))
    C_std = np.loadtxt(file_prefix + "_C_std.csv", delimiter=",")
    V = np.array(np.loadtxt(file_prefix + "_V.csv", delimiter=","))
    V_std = np.loadtxt(file_prefix + "_V_std.csv", delimiter=",")
    meta = pd.read_csv(file_prefix + "_meta.csv", delimiter=",", index_col=0)
    return C, C_std, V, V_std, meta

def weighted_var(M, W):
    avg = np.average(M, axis=1, weights=W)
    diff = ((M.T-avg).T)**2
    var = np.average(diff, axis=1, weights=W)
    return var

def weighted_errors(M, M_recov, W):
    E = np.sum(((M_recov - M) ** 2) * W)
    return E

def diff_score(C):
    from itertools import combinations
    test_list = np.arange(C.shape[0])
    res = list(combinations(test_list, 2))
    diff = []
    for i in res:
        diff.append(np.abs(np.diff(C[[i]][0], axis=0))[0])
    diff = np.array(diff)
    return np.mean(np.mean(diff, axis=0))





def get_best_run(run_prefix, raw_prefix, parfile):
    best_score, best_line = 0, -1
    C, C_std, V, V_std, meta = None, None, None, None, None

    for i in np.arange(1, 7):
        filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, i)
        file_prefix = run_prefix + "_" + str(filter_germline) + "_" + str(filter_ref_alt_cells)
        # print(file_prefix + "_C.csv", is_non_zero_file(file_prefix + "_C.csv"))
        if is_non_zero_file(file_prefix + "_C.csv"):
            C_, C_std_, V_, V_std_, _ = get_run_data(file_prefix)
            if V_.shape[0] > 100:
                REF, ALT, meta_ = get_raw_data(raw_prefix, filter_germline, filter_ref_alt_cells)
                M = get_varcall(np.array(REF), np.array(ALT))
                cov = np.array(REF+ALT)
                W = get_weights(cov, M)
                cost =  np.sum(((np.dot(V_, C_) - M) ** 2) * W)/np.sum(W)
                new_score = diff_score(C_)
                M_var = weighted_var(M, W)
                top_n = np.flip(np.argsort(M_var))
                perc_var = np.round(np.sum(weighted_var(np.dot(V_, C_), W))/np.sum(weighted_var(M, W))*100,1)
                

                if best_score < new_score:
                    C, C_std, V, V_std, meta = C_, C_std_, V_, V_std_, meta_
                    best_score = new_score
                    best_line = i

    readme = (best_line, best_score)
    if best_line>-1:
        filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, best_line)
        REF, ALT, _ = get_raw_data(raw_prefix, filter_germline, filter_ref_alt_cells)
        cell_assignments = pd.DataFrame(C.T, index=ALT.columns)
    else:
        REF, ALT, cell_assignments = None, None, None
    
    return cell_assignments, C_std, V, V_std, REF, ALT, meta, readme


def get_best_k(curve):
    import numpy.matlib
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    print(lineVec)
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    print(lineVecNorm)
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    print(vecToLine)
    distToLine = np.sqrt(np.sum(vecToLine, axis=1))
    print(distToLine)
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def get_best_run_multik(run_prefix, patient, raw_prefix, parfile):
    
    best_score, best_line = 0, -1
    all_C, all_C_std, all_V, all_V_std, all_meta = [], [], [], [], []
    all_REF, all_ALT = [], []
    all_per, all_sc = [], []
    runned_subsets = []
    
    
    for i in np.arange(1, 7):
        filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, i)
        REF, ALT, meta = get_raw_data(raw_prefix, filter_germline, filter_ref_alt_cells)
        # print(REF.shape[0])
        
        if REF.shape[0]>100:
            runned_subsets.append(i)
            M = get_varcall(np.array(REF), np.array(ALT))
            W = get_weights(np.array(REF+ALT), M)
            var = weighted_var(M, W)
            
            
            i_C, i_C_std, i_V, i_V_std = [], [], [], []
            i_per, i_sc = [], []
            
            try:
                
                for k in np.arange(1, 5):
                    file_prefix = run_prefix + "k"+str(k)+"_" + patient + "_" + str(filter_germline) + "_" + str(filter_ref_alt_cells)
                    C, C_std, V, V_std, _ = get_run_data(file_prefix)
                    C, V = (np.array([C]), np.array([V]).T) if k == 1 else (C, V)
                    M_recovered = np.dot(V, C)# if k>1 else np.dot(np.array(V).T, C)
                    E = weighted_errors(M, M_recovered, W)
                    sc = diff_score(C) if k>1 else 0
                    i_C.append(C), i_C_std.append(C_std), i_V.append(V), i_V_std.append(V_std)
                    i_per.append(E), i_sc.append(sc)
                
                
                all_C.append(i_C), all_C_std.append(i_C_std), all_V.append(i_V), all_V_std.append(i_V_std)
                i_per_scaled = (i_per-np.min(i_per))/(np.max(i_per)-np.min(i_per))
                all_per.append(i_per_scaled), all_sc.append(i_sc)
                all_REF.append(REF), all_ALT.append(ALT), all_meta.append(meta)
                plt.scatter(np.arange(1, 5), i_per_scaled, alpha=.5, color="grey")
                    
            except:
                print("missing file for "+str(i))
            


    plt.scatter(np.arange(1, 5), np.mean(all_per, axis=0))
    plt.show()
    all_sc = np.array(all_sc)
    best_k = get_best_k(np.mean(all_per, axis=0))#/np.max(all_per, axis=0))

    best_i = np.argmax(all_sc[:,best_k])
    
    print(best_i, best_k)
                
    #readme = (best_line, best_score)    
    readme = (runned_subsets[best_i], all_sc[best_i, best_k])
    cell_assignments = pd.DataFrame(all_C[best_i][best_k].T, index=ALT.columns)
    to_return = {"cell_assignments": cell_assignments, "C_std": all_C_std[best_i][best_k], "V": all_V[best_i][best_k], "V_std": all_V_std[best_i][best_k], "REF": all_REF[best_i], "ALT": all_ALT[best_i],
                 "meta": all_meta[best_i], "readme": readme}
    return to_return
