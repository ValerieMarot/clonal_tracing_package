import numpy as np
from scipy.stats import pearsonr


def filter_vars(REF, ALT, meta,
                filter_germline=True,
                filter_RNA_edits=True,
                filter_ref_alt_cells=5):
    f_r_a_c = (filter_ref_alt_cells / 100) * ALT.shape[1]
    # only keep variants seen and NOT seen in at least min_ref_alt_cells cells
    # i.e. kick out variants seen in almost all or almost no cells
    only_REF = np.sum((REF >= 2) & (ALT <= 2), axis=1)
    only_ALT = np.sum((REF <= 2) & (ALT >= 2), axis=1)
    REF_and_ALT = np.sum((REF >= 2) & (ALT >= 2), axis=1)
    sub = (only_ALT >= f_r_a_c) & (only_REF >= f_r_a_c)
    sub |= (only_ALT >= f_r_a_c) & (REF_and_ALT >= f_r_a_c)
    sub |= (only_REF >= f_r_a_c) & (REF_and_ALT >= f_r_a_c)
    # only keep variants covered in at least min_cov_cells  cells
    if filter_germline:  # filter germline variants
        sub &= ~meta.dbSNP
    if filter_RNA_edits:  # filter RNA edits
        sub &= ~meta.REDIdb
    # subset
    sub |= meta.chr == "chrM"

    print("Filtering \t" + str(np.sum(~sub)) + " variants.")
    print("Keeping \t" + str(np.sum(sub)) + " variants.")
    REF, ALT = REF.loc[sub], ALT.loc[sub]
    meta = meta.loc[sub]
    return REF, ALT, meta


def filter_vars_from_same_read(REF, ALT, meta, dist=np.infty, pearson_corr=.95, merge_WE=False):
    VAF = REF / (REF + ALT)

    p = np.array(meta.pos)
    neighbor = np.where((p[1:] - p[:-1]) < dist)[0]
    idx = meta.index[neighbor]
    idx_ = meta.index[neighbor + 1]

    c = 0
    print("comparing " + str(len(neighbor)))

    for j in range(len(idx)):
        i, i_ = idx[j], idx_[j]
        x, y = VAF.loc[i], VAF.loc[i_]
        sub = ~(np.isnan(x) | np.isnan(y))
        if np.sum(sub) > 20:
            val = pearsonr(x[sub], y[sub])[0]
            if val > pearson_corr:
                REF, ALT, meta = merge_row(i, i_, REF, ALT, meta, merge_WE=merge_WE)
                c += 1

    print("filtered " + str(c))
    return REF, ALT, meta


def merge_row(i, i_, REF, ALT, meta, merge_WE):
    row = meta.loc[i_]
    rows = meta.loc[[i, i_]]
    # merge meta
    row.pos = "_".join(np.array(rows.pos).astype(str))
    row.dnSNP, row.REDIdb = np.any(rows.dbSNP), np.any(rows.REDIdb)

    row.ref = "_".join(np.array(rows.ref))
    row.mut = "_".join(np.array(rows.mut))
    if merge_WE:
        row.cancer_ref, row.cancer_alt = np.sum(rows.cancer_ref), np.sum(rows.cancer_alt)
        row.healthy_ref, row.healthy_alt = np.sum(rows.healthy_ref), np.sum(rows.healthy_alt)
    REF.loc[i_] = np.sum(REF.loc[[i_, i]], axis=0)
    ALT.loc[i_] = np.sum(ALT.loc[[i_, i]], axis=0)

    # todo merge gene entry

    meta.loc[i_] = row
    meta, REF, ALT = meta.drop(i, axis=0), REF.drop(i, axis=0), ALT.drop(i, axis=0)
    return REF, ALT, meta


def get_pars_from_line(file, line_number):
    f = open(file)
    lines = f.readlines()
    pars = lines[line_number - 1].split(",")
    filter_germline, filter_ref_alt_cells = pars[0] == "True", int(pars[1])
    return filter_germline, filter_ref_alt_cells
