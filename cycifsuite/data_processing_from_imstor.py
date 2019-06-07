import pandas as pd
import os
import random
import numpy as np
# setting path
os.chdir('Z:/sorger/data/IN_Cell_Analyzer_6000/Connor/Fixed MCF10 Common/20x full exp/20180905_Updated/')
path_out = 'N:/HiTS Projects and Data/Personal/Jake/mcf10a/'
# get list of valid columns and feature selected columns
feature_metadata = pd.read_json(path_out + 'feature_metadata.json')
feature_metadata = feature_metadata.transpose()
valid_cols = feature_metadata.index
selected_cols = feature_metadata[feature_metadata.feature_type.isin(
    ['mean', 'median', 'standev', 'lawsls5', 'lawsls9'])].index
# plate information for experiments
plate_info = pd.read_csv(
    'd:/data/MCF10A 090718 data/plate_key.csv', index_col=0)
plate_info.index = plate_info.index.astype(str)

# Iterate through each plate, well and field, aka, pwf
metadata = pd.DataFrame()
for plate in range(1, 19):
    expr_data = pd.DataFrame()
    plate_id = plate
    plate = 'plate' + str(plate)
    print(plate)
    # Assumes the data was organized by plate and results are in the /analysis
    # folder
    path_analysis = os.path.join(plate, 'analysisCorrected')
    txt_files = [x for x in os.listdir(path_analysis) if 'txt' in x]
    for txt_fn in txt_files:
        well = txt_fn.split('_')[0]
        field = txt_fn.split('_')[1]
        cell_name_prefix = '_'.join([str(plate_id), well, field])
        fn = os.path.join(path_analysis, txt_fn)
        _df = pd.read_table(fn)
        _df.index = [cell_name_prefix + '_' +
                     str(x) for x in range(1, 1 + _df.shape[0])]
        # make metadata for the current pwf
        _metadata = pd.DataFrame(index=_df.index)
        _metadata['field'] = field
        _metadata['Plate'] = plate_id
        _metadata['pw'] = '{}_{}'.format(plate_id, well)
        expr_data = expr_data.append(_df[valid_cols])
        metadata = metadata.append(_metadata)

    # writing data out in two formats, full feature format and selected
    # features.
    raw_data_fn = ''.join([path_out, 'raw_data_with_txt_features/plate' +
                           str(plate_id).zfill(2), '_proper_FFC_expr_data.hdf'])
    fs_data_fn = ''.join([path_out, 'raw_data_selected_features/plate' +
                          str(plate_id).zfill(2), '_proper_FFC_expr_data.hdf'])
    expr_data.to_hdf(raw_data_fn, plate, mode='w')
    expr_data[selected_cols].to_hdf(fs_data_fn, plate, mode='w')

# complete metadata with experiment info.
plate_info.columns = ['Well', 'ligand', 'dose', 'time']
plate_info['pw'] = plate_info.index.astype(str) + '_' + plate_info.Well
metadata['ori_idx'] = metadata.index
metadata = metadata.merge(
    plate_info, on='pw', how='left').set_index('ori_idx')
for key, value in {'t0_ctrl': 'ctrl', 'control': 'PBS', 'TGFb': 'TGFB', 'IFNg': 'IFNG'}.items():
    metadata.loc[metadata.ligand == key, 'ligand'] = value

# additional metadata required by OHSU
metadata['replicate'] = 'C'
metadata.loc[metadata.Plate <= 12, 'replicate'] = 'B'
metadata.loc[metadata.Plate <= 6, 'replicate'] = 'A'
metadata.loc[metadata.dose == 't0_ctrl', 'dose'] = 0
metadata['condition'] = metadata.ligand + '_' + metadata.time.astype(str)
metadata['collection'] = 'C3'
metadata['sampleid'] = metadata[['ligand', 'time', 'collection',
                                 'replicate']].astype(str).apply(lambda x: '_'.join(x), axis=1)
metadata.to_csv(path_out + 'proper_FFC_metadata.csv')
