import numpy as np
import scipy


def filter_vars(REF, ALT, meta,
                VAF_thres=.2,
                filter_germline=True, filter_RNA_edits=True,
                filter_artefacts_thres=0,
                filter_ref_alt_cells=5):
    f_r_a_c = (filter_ref_alt_cells / 100) * ALT.shape[1]
    # only keep variants seen and NOT seen in at least min_ref_alt_cells cells
    # i.e. kick out variants seen in almost all or almost no cells
    sub = (np.sum(((ALT / (REF + ALT)) > VAF_thres) & (ALT >= 2), axis=1) >= f_r_a_c)
    sub &= (np.sum(((REF / (REF + ALT)) > VAF_thres) & (REF >= 2), axis=1) >= f_r_a_c)
    # only keep variants covered in at least min_cov_cells  cells
    if filter_germline:  # filter germline variants
        sub &= ~meta.dbSNP
    if filter_RNA_edits:  # filter RNA edits
        sub &= ~meta.REDIdb
    if filter_artefacts_thres != 0:  # filter variants thought to be artefacts
        if filter_artefacts_thres > 0:
            sub &= (meta["NN_pred_real"] > filter_artefacts_thres)
        else:
            sub &= (meta["NN_pred_real"] < -filter_artefacts_thres)
    print("Filtering \t" + str(np.sum(~sub)) + " variants.")
    print("Keeping \t" + str(np.sum(sub)) + " variants.")
    # subset
    REF, ALT = REF.loc[sub], ALT.loc[sub]
    meta = meta.loc[sub]
    return REF, ALT, meta


def filter_vars_from_same_read(REF, ALT, meta, dist=100, pearson_corr=.95):
    VAF = REF / (REF + ALT)

    p = meta.pos
    neighbor = np.where((p[1:] - p[:-1] < dist))[0]
    idx = meta.index[neighbor]
    idx_ = meta.index[neighbor + 1]

    for j in range(len(idx)):
        i, i_ = idx[j], idx_[j]
        x, y = VAF.loc[i], VAF.loc[i_]
        sub = ~(np.isnan(x) | np.isnan(y))
        if np.sum(sub) > 20:
            if np.abs(scipy.stats.pearsonr(x[sub], y[sub])[0]) > pearson_corr:
                meta, REF, ALT = merge_row(i, i_, meta, REF, ALT)
    return meta, REF, ALT


def merge_row(i, i_, meta, REF, ALT):
    row = meta.loc[i_]
    row.pos = "_".join(np.array(meta.loc[[i, i_]].pos).astype(str))
    row.ref = "_".join(np.array(meta.loc[[i, i_]].ref))
    row.mut = "_".join(np.array(meta.loc[[i, i_]].mut))
    row.dnSNP = np.any(meta.loc[[i, i_]].dbSNP)
    row.REDIdb = np.any(meta.loc[[i, i_]].REDIdb)
    row.cancer_ref = np.sum(meta.loc[[i, i_]].cancer_ref)
    row.cancer_alt = np.sum(meta.loc[[i, i_]].cancer_alt)
    row.cancer_cov = np.sum(meta.loc[[i, i_]].cancer_cov)
    row.healthy_ref = np.sum(meta.loc[[i, i_]].healthy_ref)
    row.healthy_alt = np.sum(meta.loc[[i, i_]].healthy_alt)
    row.healthy_cov = np.sum(meta.loc[[i, i_]].healthy_cov)
    meta.loc[i_] = row

    REF.loc[i_] = np.sum(REF.loc[[i_, i]])
    ALT.loc[i_] = np.sum(ALT.loc[[i_, i]])

    meta, REF, ALT = meta.drop(i, axis=0), REF.drop(i, axis=0), ALT.drop(i, axis=0)
    return meta, REF, ALT


def get_pars_from_line(file, line_number):
    f = open(file)
    lines = f.readlines()
    pars = lines[line_number - 1].split(",")
    filter_germline, filter_ref_alt_cells = pars[0] == "True", int(pars[1])
    return filter_germline, filter_ref_alt_cells
