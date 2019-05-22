from cycifsuite.detect_lost_cells import ROC_lostcells, get_lost_cells, plot_lost_cell_per_cycle_stacked_area
import pandas as pd

raw_qc_expr_data = pd.read_csv(
    'example_input/sample_raw_data_for_QC.csv', index_col=0)
raw_qc_expr_data = raw_qc_expr_data.iloc[:, :9]
# cycle difference thresholding
_, _, t_c = ROC_lostcells(raw_qc_expr_data.iloc[:, 1:], cutoff_min=0, cutoff_max=1,
                          steps=50, filtering_method='cycle_diff', figname='example_output/Cycle difference thresholding.png')
# background thresholding
_, _, t_bgnd = ROC_lostcells(raw_qc_expr_data, cutoff_min=0, cutoff_max=5,
                             steps=50, figname='example_output/Background thresholding.png')
# get lost cells.
df_lc_cv, _ = get_lost_cells(
    raw_qc_expr_data.iloc[:, 1:], t_c, 8, 'cycle_diff')
# ploting lost cells by fields
plot_lost_cell_per_cycle_stacked_area(
    df_lc_cv, figname='example_output/accumulated cell loss by cycle.png')
