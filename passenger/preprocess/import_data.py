import pandas as pd
import numpy as np
from passenger.preprocess.import_sequence_context import get_variant_as_matrix, is_non_zero_file
from scipy.io import mmread


def get_meta(meta_file, annotation_file, chrom, NN_model, path_to_context_data, path_to_exome_data,
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

    # filter
    print(meta_0.shape)
    REDIdb = np.ones(meta_0.shape[0])
    dbSNP = np.ones(meta_0.shape[0])
    gene = np.repeat("", meta_0.shape[0])
    for i, pos in enumerate(meta_0.pos):
        idx = np.where(ann_0["pos"] == pos)[0]
        REDIdb[i] = np.any(ann_0.iloc[idx]["REDIdb"])
        dbSNP[i] = np.any(ann_0.iloc[idx]["dbSNP-common"])
        gene[i] = ",".join(np.unique(ann_0.iloc[idx]["gene"]))

    meta_0["REDIdb"] = REDIdb
    meta_0["dbSNP"] = dbSNP
    meta_0["gene"] = gene

    if NN_model is not None:
        rows = []
        for i in range(meta_0.shape[0]):
            entry = meta_0.iloc[i]
            data = get_variant_as_matrix(entry["chr"], entry["pos"], window=30, path=path_to_context_data)
            rows.append(data)
        rows = np.array(rows)
        pred = NN_model.predict(rows)
        meta_0["NN_pred_real"] = pred[:, 1]
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


def load_NN_model(model):
    from tensorflow import keras

    model = keras.models.load_model(model)
    prob_model = keras.models.Sequential([model, keras.layers.Softmax()])
    return prob_model


def get_variant_measurement_data(path,
                                 all_chroms=None,
                                 cell_names=None,
                                 NN_filter_artefacts_path=None,
                                 path_to_context_data="",
                                 path_to_exome_data=None,
                                 datatype="S2",
                                 sub_cell_names=None):
    meta, ann, ALT, REF = None, None, None, None
    all_chroms = ["chr" + str(i) for i in range(1, 23)] if all_chroms is None else all_chroms

    if NN_filter_artefacts_path is not None:
        NN_model = load_NN_model(NN_filter_artefacts_path)
    else:
        NN_model = None

    for chrom in all_chroms:
        print(chrom)
        skip = False
        if datatype == "S2":
            ALT_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-ALT.csv", index_col=False, header=None)
            REF_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-REF.csv", index_col=False, header=None)
            meta_0 = get_meta(path + "vcf/processed-" + chrom + "-meta.csv", path + "vcf/annotations-" + chrom + ".tsv",
                              chrom, NN_model,
                              path_to_context_data=path_to_context_data,
                              path_to_exome_data=path_to_exome_data, datatype=datatype)
        else:
            # try:
            f = open(path + '/' + chrom + '/cellSNP.tag.AD.mtx', 'r')
            ALT_0 = pd.DataFrame(mmread(f).A)

            f = open(path + '/' + chrom + '/cellSNP.tag.DP.mtx', 'r')
            DP = pd.DataFrame(mmread(f).A)

            # f = open(path+'/'+chrom+'cellSNP.tag.OTH.mtx', 'r')
            # OTH = mmread(f).A

            REF_0 = DP - ALT_0

            meta_0 = get_meta(path + '/' + chrom + '/cellSNP.base.vcf', path + '/' + chrom + '/annotations.tsv',
                              chrom, NN_model,
                              path_to_context_data=path_to_context_data,
                              path_to_exome_data=path_to_exome_data,
                              datatype=datatype)
            # except:
            print("BROKEN CHROM " + chrom + "\n\n")
            skip = True

        # variants need to be covered in at least 10% of the cells
        sub = np.sum((ALT_0 + REF_0) >= 2, axis=1) > (ALT_0.shape[1] / 10)
        ALT_0, REF_0, meta_0 = ALT_0[sub], REF_0[sub], meta_0[sub]
        if not skip:
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
