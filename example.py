import pandas as pd
import os
from plate_based_analysis import per_well_analysis

expr_data = pd.read_csv(
    'example_input/sample_exprdata.csv', index_col=0)
metadata = pd.read_csv(
    'example_input/sample_metadata.csv', index_col=0)
if not os.path.exists('example_output'):
    os.makedirs('example_output')
print('Metadata should be in a format like this:')
print(metadata.head())
control_name = 'ctrl'
output_prefix = 'T0_ctrl as control'
pwa = per_well_analysis(expr_data, metadata, batch_col='replicate',
                        drug_col='ligand', dose_col='dose', time_col='time', control_name=control_name,
                        output_path='example_output', output_prefix=output_prefix)
pwa.make_contrast(batch_wise=True)
pwa.descriptive_analysis()
pwa.contrast_based_distance()
pwa.contrast_based_differential_analysis()
pwa.plot_dist_violinplot('ligand', 'time')
pwa.make_fc_heamtap_matrix(
    ['ligand', 'time', 'Batch'], output_fname=output_prefix)
pwa.make_heatmaps(style='simple')
pwa.make_dotted_heatmap()
pwa.make_swarm_plot()
