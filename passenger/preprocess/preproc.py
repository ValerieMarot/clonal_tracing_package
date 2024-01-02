from passenger.preprocess.import_data import get_variant_measurement_data
from passenger.preprocess.filter import filter_vars_from_same_read
import numpy as np
import anndata

def preproc_and_save(in_path, 
                     cell_names, 
                     path_to_exome_data=None, perc=10,
                     filter_hela=True, out_path=None):
    REF, ALT, meta = get_variant_measurement_data(in_path,
                                                  path_to_exome_data=path_to_exome_data,
                                                  cell_names=cell_names, perc=perc,
                                                  filter_hela=filter_hela)
    REF, ALT, meta = filter_vars_from_same_read(REF, ALT, meta, dist=np.infty,
                                                merge_WE=True if path_to_exome_data is not None else False)
    meta.pos = meta.pos.astype(str)

    adata = anndata.AnnData(X=(REF+ALT).T, 
                    layers={"REF":REF.T, "ALT":ALT.T},
                    var=meta)
    if out_path is not None:
        adata.write(filename=out_path)
    return adata
