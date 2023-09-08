import numpy as np


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


def get_pars_from_line(file, line_number):
    f = open(file)
    lines = f.readlines()
    pars = lines[line_number - 1].split(",")
    filter_germline, filter_ref_alt_cells = pars[0] == "True", int(pars[1])
    return filter_germline, filter_ref_alt_cells