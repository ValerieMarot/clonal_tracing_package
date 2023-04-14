import pandas as pd
import numpy as np


def get_variant_measurement_data(path,
                                 all_chroms=None,
                                 get_annotation=True,
                                 cell_names=None,
                                 min_ref_cells=20,
                                 min_alt_cells=20,
                                 min_vaf=.2):
    meta, ann, ALT, REF = None, None, None, None
    all_chroms = ["chr" + str(i) for i in range(1, 23)] if all_chroms is None else all_chroms

    for chrom in all_chroms:

        ALT_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-ALT.csv", index_col=False, header=None)
        REF_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-REF.csv", index_col=False, header=None)
        meta_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-meta.csv", index_col=False)
        sub = np.ones(ALT_0.shape[0]).astype(bool)

        if get_annotation:
            ann_0 = pd.read_csv(path + "vcf/annotations-" + chrom + ".tsv", sep="\t", index_col=False, header=None,
                                comment="#")
            ann_0.insert(0, 'chr', np.repeat(chrom, ann_0.shape[0]))
            ann_0.columns = ["chr", "pos", "allele", "gene", "feature", "feature_type", "consequence", "AA", "Codons",
                             "Existing_variation", "effect", "REDIdb", "AF", "dbSNP-common"]

            # filter
            ann_0["pos"] = [i.split(":")[1] for i in ann_0["pos"]]
            to_drop = (ann_0["REDIdb"] != "-")
            pos = np.unique(ann_0.loc[to_drop]["pos"].tolist()).astype('int64')
            sub_ = [i in pos for i in meta_0.pos]
            meta_0["REDIdb"] = sub_

            to_drop |= (ann_0["dbSNP-common"] != "-")
            pos = np.unique(ann_0.loc[to_drop]["pos"].tolist()).astype('int64')
            sub_ = [i in pos for i in meta_0.pos]
            meta_0["dbSNP"] = sub_

        ALT_VAF = (ALT_0 / (REF_0 + ALT_0))
        sub &= (np.sum((ALT_VAF > min_vaf) & (ALT_0 >= 2), axis=1) >= min_alt_cells)  # at least x variant cells
        sub &= (np.sum((ALT_VAF < min_vaf) & (REF_0 >= 2), axis=1) >= min_ref_cells)
        print(chrom + " - " + str(np.sum(np.array(sub))))
        meta_0 = meta_0[sub]
        REF_0, ALT_0 = REF_0[sub], ALT_0[sub]

        ALT = ALT_0 if ALT is None else pd.concat((ALT, ALT_0))
        REF = REF_0 if REF is None else pd.concat((REF, REF_0))
        meta = meta_0 if meta is None else pd.concat((meta, meta_0))
        if get_annotation:
            ann = ann_0 if ann is None else pd.concat((ann, ann_0))

    # convert
    last = REF.shape[1] - 1
    REF = REF.drop(last, axis=1)
    ALT = ALT.drop(last, axis=1)

    # reset indices
    REF.index = np.arange(0, REF.shape[0])
    ALT.index = np.arange(0, REF.shape[0])
    meta.index = np.arange(0, REF.shape[0])
    if get_annotation:
        ann.index = np.arange(0, ann.shape[0])

    if cell_names is not None:
        REF.columns, ALT.columns = cell_names, cell_names

    # additional annotation
    meta["REF"] = np.sum(REF > 0, axis=1)
    meta["ALT"] = np.sum(ALT > 0, axis=1)
    meta["REF_cov"] = np.sum(REF, axis=1)
    meta["ALT_cov"] = np.sum(ALT, axis=1)

    return ALT, REF, meta, ann
