import pandas as pd
import numpy as np
# from passenger.preprocess.import_sequence_context import get_variant_as_matrix, is_non_zero_file
from scipy.io import mmread
import os


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def get_meta(meta_file, annotation_file, chrom, path_to_exome_data,
             datatype):
    if datatype == "S2":
        meta_0 = pd.read_csv(meta_file, index_col=False)
    else:
        meta_0 = pd.read_csv(meta_file, index_col=False, header=None,
                             sep="\t", comment="#")
        meta_0.columns = ["chr", "pos", "ID", "ref", "mut", "qual", "filter", "info"]
        meta_0 = meta_0[["chr", "pos", "ref", "mut"]]
    ann_0 = pd.read_csv(annotation_file, sep="\t", index_col=False, header=None, comment="#")
    ann_0.insert(0, 'chr', np.repeat(chrom, ann_0.shape[0]))
    ann_0.columns = ["chr", "pos", "allele", "gene", "feature", "feature_type", "consequence", "AA", "Codons",
                     "Existing_variation", "effect", "REDIdb", "AF", "dbSNP-common"]
    ann_0["pos"] = [int(i.split(":")[1]) for i in ann_0["pos"]]
    ann_0["cons"] = ann_0["gene"] + ":" + ann_0["consequence"]
    # filter
    REDIdb = np.ones(meta_0.shape[0]).astype(bool)
    dbSNP = np.ones(meta_0.shape[0]).astype(bool)
    gene = np.repeat("", meta_0.shape[0]).astype(str)
    for i, pos in enumerate(meta_0.pos):
        idx = np.where(ann_0["pos"] == pos)[0]
        REDIdb[i] = np.any(ann_0.iloc[idx]["REDIdb"] != "-")
        dbSNP[i] = np.any(ann_0.iloc[idx]["dbSNP-common"] != "-")
        gene[i] = ",".join(np.unique(ann_0.iloc[idx]["cons"]))

    meta_0["REDIdb"], meta_0["dbSNP"], meta_0["gene"] = REDIdb, dbSNP, gene

    if path_to_exome_data is not None:
        cancer_file = path_to_exome_data + "cancer-filtered-var-pileup-" + chrom + ".vcf"
        healthy_file = path_to_exome_data + "healthy-filtered-var-pileup-" + chrom + ".vcf"
        if is_non_zero_file(cancer_file) and is_non_zero_file(healthy_file):
            cancer_ref, cancer_alt, cancer_cov = get_WE_data(cancer_file, meta_0)
            healthy_ref, healthy_alt, healthy_cov = get_WE_data(healthy_file, meta_0)
        meta_0["cancer_ref"] = cancer_ref
        meta_0["cancer_alt"] = cancer_alt
        meta_0["cancer_cov"] = cancer_cov
        meta_0["healthy_ref"] = healthy_ref
        meta_0["healthy_alt"] = healthy_alt
        meta_0["healthy_cov"] = healthy_cov
    return meta_0


def get_variant_measurement_data(path,
                                 all_chroms=None,
                                 cell_names=None,
                                 path_to_exome_data=None,
                                 datatype="S2",
                                 sub_cell_names=None):
    meta, ann, ALT, REF = None, None, None, None
    all_chroms = ["chr" + str(i) for i in range(1, 23)] if all_chroms is None else all_chroms

    for chrom in all_chroms:
        print(chrom)
        if datatype == "S2":
            ALT_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-ALT.csv", index_col=False, header=None)
            REF_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-REF.csv", index_col=False, header=None)
            meta_0 = get_meta(path + "vcf/processed-" + chrom + "-meta.csv", path + "vcf/annotations-" + chrom + ".tsv",
                              chrom,
                              path_to_exome_data=path_to_exome_data, datatype=datatype)
        else:
            f = open(path + '/' + chrom + '/cellSNP.tag.AD.mtx', 'r')
            ALT_0 = pd.DataFrame(mmread(f).A)

            f = open(path + '/' + chrom + '/cellSNP.tag.DP.mtx', 'r')
            DP = pd.DataFrame(mmread(f).A)

            REF_0 = DP - ALT_0

            meta_0 = get_meta(path + '/' + chrom + '/cellSNP.base.vcf', path + '/' + chrom + '/annotations.tsv',
                              chrom,
                              path_to_exome_data=path_to_exome_data,
                              datatype=datatype)

        # variants need to be covered in at least 10% of the cells
        sub = np.sum((ALT_0 + REF_0) >= 2, axis=1) > (ALT_0.shape[1] / 10)
        ALT_0, REF_0, meta_0 = ALT_0[sub], REF_0[sub], meta_0[sub]

        print(meta_0.shape)

        ALT = ALT_0 if ALT is None else pd.concat((ALT, ALT_0))
        REF = REF_0 if REF is None else pd.concat((REF, REF_0))
        meta = meta_0 if meta is None else pd.concat((meta, meta_0))

    if datatype == "S2":  # convert
        last = REF.shape[1] - 1
        REF = REF.drop(last, axis=1)
        ALT = ALT.drop(last, axis=1)

    # reset indices
    REF.index = np.arange(0, REF.shape[0])
    ALT.index = np.arange(0, REF.shape[0])
    meta.index = np.arange(0, REF.shape[0])

    if cell_names is not None:
        REF.columns, ALT.columns = cell_names, cell_names
        if sub_cell_names is not None:
            REF = REF[sub_cell_names]
            ALT = ALT[sub_cell_names]
    int_pos = np.array([int(i) for i in meta.pos.tolist()])
    in_region = (meta.chr == "chr6") & (int_pos > 28510120) & (int_pos < 33480577)
    print(np.sum(in_region), " vars in HLA regions")
    sub = ~in_region
    REF, ALT = REF.loc[sub], ALT.loc[sub]
    meta = meta.loc[sub]
    return REF, ALT, meta


def get_WE_data(path_to_file, meta_):
    WE_cancer = pd.read_csv(path_to_file,
                            comment="#", sep="\t", index_col=False, header=None)
    ref, alt, cov = [], [], []

    for i in range(meta_.shape[0]):
        meta_pos = WE_cancer[1] == meta_.iloc[i].pos
        if not np.any(meta_pos):
            ref.append(0), alt.append(0), cov.append(0)
            continue
        else:
            WE_c_entry = WE_cancer.loc[meta_pos].iloc[0]
            vals = WE_c_entry[9].split(":")[-1].split(",")
            alleles = WE_c_entry[4].split(",")
            ref.append(int(vals[0]))
            cov.append(np.sum(np.array(vals).astype(int)))
            vals = vals[1:]
            found = False
            for j, a in enumerate(alleles):
                if a == meta_.iloc[i].mut:
                    alt.append(int(vals[j]))
                    found = True
            if not found:
                alt.append(0)
    return ref, alt, cov
