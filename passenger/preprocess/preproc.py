from passenger.preprocess.import_data import get_variant_measurement_data
from passenger.preprocess.filter import *
import numpy as np


def preproc_and_save(in_path, cell_names, datatype, merge_WE, path_to_exome_data,
                     out_path):
    REF, ALT, meta = get_variant_measurement_data(in_path,
                                                  path_to_exome_data=path_to_exome_data,
                                                  datatype=datatype,
                                                  cell_names=cell_names)
    REF, ALT, meta = filter_vars_from_same_read(REF, ALT, meta, dist=np.infty, merge_WE=merge_WE)
    ALT.to_csv(out_path + "/ALT.csv"), REF.to_csv(out_path + "/REF.csv"), meta.to_csv(out_path + "/meta.csv")


