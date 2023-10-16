from passenger.preprocess.import_data import get_variant_measurement_data
from passenger.preprocess.filter import filter_vars_from_same_read
import numpy as np


def preproc_and_save(in_path, out_path, cell_names, datatype, path_to_exome_data=None):
    REF, ALT, meta = get_variant_measurement_data(in_path,
                                                  path_to_exome_data=path_to_exome_data,
                                                  datatype=datatype,
                                                  cell_names=cell_names)
    REF, ALT, meta = filter_vars_from_same_read(REF, ALT, meta, dist=np.infty,
                                                merge_WE=True if path_to_exome_data is not None else False)
    ALT.to_csv(out_path + "_ALT.csv"), REF.to_csv(out_path + "_REF.csv"), meta.to_csv(out_path + "_meta.csv")
