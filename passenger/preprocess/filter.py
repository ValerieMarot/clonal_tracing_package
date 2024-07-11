import numpy as np
from scipy.stats import pearsonr


def filter_vars(adata,
                filter_germline=True,
                filter_RNA_edits=True,
                min_MAF=5):
    f_r_a_c = (min_MAF / 100) * adata.X.shape[0]
    # only keep variants seen and NOT seen in at least min_ref_alt_cells cells
    # i.e. kick out variants seen in almost all or almost no cells
    cov = adata.X
    REF, ALT = adata.layers["REF"], adata.layers["ALT"]
    only_REF = np.sum((REF >= 2) & (ALT == 0), axis=0)
    only_ALT = np.sum((REF == 0) & (ALT >= 2), axis=0)
    any_REF = np.sum((REF >= 2) & (REF/cov>.3), axis=0)
    any_ALT = np.sum((ALT >= 2) & (ALT/cov>.3), axis=0)
    sub = (only_ALT >= f_r_a_c) & (any_REF >= f_r_a_c)
    sub |= (only_REF >= f_r_a_c) & (any_ALT >= f_r_a_c)
    # only keep variants covered in at least min_cov_cells  cells
    if filter_germline:  # filter germline variants
        sub &= ~adata.var.dbSNP
    if filter_RNA_edits:  # filter RNA edits
        sub &= ~adata.var.REDIdb
    # subset
    sub |= adata.var.chr == "chrM"

    adata = adata[:, sub]

    adata.uns["filter_germline"] = filter_germline
    adata.uns["min_MAF"] = min_MAF

    return adata


def filter_vars_from_same_read(REF, ALT, meta, dist=np.infty, pearson_corr=.95):
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
                REF, ALT, meta = merge_row(i, i_, REF, ALT, meta)
                c += 1

    print("filtered " + str(c))
    return REF, ALT, meta


def merge_row(i, i_, REF, ALT, meta):
    row = meta.loc[i_]
    rows = meta.loc[[i, i_]]
    # merge meta
    row.pos = "_".join(np.array(rows.pos).astype(str))
    row.dnSNP, row.REDIdb = np.any(rows.dbSNP), np.any(rows.REDIdb)

    row.ref = "_".join(np.array(rows.ref))
    row.mut = "_".join(np.array(rows.mut))

    REF.loc[i_] = np.sum(REF.loc[[i_, i]], axis=0)
    ALT.loc[i_] = np.sum(ALT.loc[[i_, i]], axis=0)

    meta.loc[i_] = row
    meta, REF, ALT = meta.drop(i, axis=0), REF.drop(i, axis=0), ALT.drop(i, axis=0)
    return REF, ALT, meta


def get_pars_from_line(file, line_number):
    f = open(file)
    lines = f.readlines()
    pars = lines[line_number - 1].split(",")
    filter_germline, filter_ref_alt_cells = pars[0] == "True", int(pars[1])
    return filter_germline, filter_ref_alt_cells
