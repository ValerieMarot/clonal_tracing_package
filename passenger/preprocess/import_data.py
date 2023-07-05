import pandas as pd
import numpy as np
import os
from passenger.preprocess.import_sequence_context import get_variant_as_matrix


def get_meta(meta_file, annotation_file, chrom, NN_model, path_to_conext_data):
    meta_0 = pd.read_csv(meta_file, index_col=False)
    ann_0 = pd.read_csv(annotation_file, sep="\t", index_col=False, header=None, comment="#")
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

    if NN_model is not None:
        rows = []
        # print(meta_0)
        for i in range(meta_0.shape[0]):
            entry = meta_0.iloc[i]
            prefix = path_to_conext_data + entry["chr"] + "_" + str(entry["pos"])
            if os.path.isfile(prefix + "_ALT.csv"):
                rows.append(get_variant_as_matrix(entry["chr"], entry["pos"], window=30, path=path_to_conext_data))
            else:
                rows.append(np.ones((61, 8)) * np.nan)
        rows = np.array(rows)
        print(rows)
        pred = NN_model.predict(rows)
        print(pred)
        print(pred[:,1])
        meta_0["NN_pred_real"] = pred[:, 1]
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
                                 path_to_context_data=""):
    meta, ann, ALT, REF = None, None, None, None
    all_chroms = ["chr" + str(i) for i in range(1, 23)] if all_chroms is None else all_chroms

    if NN_filter_artefacts_path is not None:
        NN_model = load_NN_model(NN_filter_artefacts_path)
    else:
        NN_model = None

    for chrom in all_chroms:
        print(chrom)
        ALT_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-ALT.csv", index_col=False, header=None)
        REF_0 = pd.read_csv(path + "vcf/processed-" + chrom + "-REF.csv", index_col=False, header=None)
        meta_0 = get_meta(path + "vcf/processed-" + chrom + "-meta.csv", path + "vcf/annotations-" + chrom + ".tsv",
                          chrom, NN_model, path_to_context_data)

        ALT = ALT_0 if ALT is None else pd.concat((ALT, ALT_0))
        REF = REF_0 if REF is None else pd.concat((REF, REF_0))
        meta = meta_0 if meta is None else pd.concat((meta, meta_0))

    # convert
    last = REF.shape[1] - 1
    REF = REF.drop(last, axis=1)
    ALT = ALT.drop(last, axis=1)

    # reset indices
    REF.index = np.arange(0, REF.shape[0])
    ALT.index = np.arange(0, REF.shape[0])
    meta.index = np.arange(0, REF.shape[0])

    if cell_names is not None:
        REF.columns, ALT.columns = cell_names, cell_names

    # variants need to be covered in at least 10% of the cells
    sub = np.sum((ALT + REF) >= 2, axis=1) > (ALT.shape[1]/10)
    ALT, REF, meta = ALT.loc[sub], REF.loc[sub], meta.loc[sub]

    return ALT, REF, meta
