import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_lost_cells(expr_DAPIs, threshold, n_cycles,
                   filtering_method='bgnd',
                   segmentation_cycle=4):
    """ find lost cells based on one of the two algorithms. If background based
    approach is used, cells with DAPI signal at segmentation cycle less than background 
    plus a threshold will be dropped. The threshold is inferred as fold if it is less 
    than 100, otherwise it is interpreted as abosolute value to be added to background.
    if cycle variation method is used, DAPI signal of each cycle will be divided by the 
    difference betweeen signal of current cycle to that of the next one. The quotient will
    be compared with the threshold. 

    Parameters
    --------
    expr_DAPIs: pd.DataFrame
        table of cells by channel/markers. First column need to be background, and the following
        columns will be aranged from cycle 1 to cycle n.
    threshold: numeric
        threshold for determining lost cells.
    n_cycles: int
        number of cycles in experiment.
    filtering_method: str
        method to determine lost cells. 'bgnd' or 'cycle_diff'. 'bgnd' for background based 
        thresholding and 'cycle_diff' for cycle variation based approach.
    segmentation_cycle: int
        cycle where segmentation occurred. 

    Returns
    --------
    lost_cells: list
        list of lost cell indices.
    """
    all_cells = expr_DAPIs.index
    lost_cells = pd.Series(index=all_cells)

    if filtering_method == 'bgnd':
        bgnd = expr_DAPIs.iloc[:, 0]
        if threshold < 100:
            threshold = threshold * bgnd
        else:
            threshold = threshold + bgnd
        current_cycle_fi = expr_DAPIs.iloc[:, segmentation_cycle]
        current_lost_cells = all_cells[(current_cycle_fi <= threshold)]
        lost_cells[current_lost_cells] = 'lost_due_to_high_bgnd'

    elif filtering_method == 'cycle_diff':
        for i in range(n_cycles, 1, -1):
            current_cycle_fi = expr_DAPIs.iloc[:, i]
            next_cycle_fi = expr_DAPIs.iloc[:, i - 1]
            cycle_diff = abs(next_cycle_fi - current_cycle_fi)
            current_lost_cells = all_cells[
                (current_cycle_fi < threshold * cycle_diff)]
            lost_cells[current_lost_cells] = 'Cycle_' + str(i)
    return lost_cells.dropna()


def plot_lost_cell_stacked_area(lost_cells, n_fields=9, n_cycles=8):
    lost_cells = pd.DataFrame(lost_cells, columns=['cycle'])
    lost_cells['field'] = [x.split('_')[-2] for x in lost_cells.index]
    lost_cells['count'] = 1
    lc_stats = lost_cells.groupby(['cycle', 'field']).sum()
    cycles = lc_stats.index.levels[0]
    flds = lc_stats.index.levels[1]
    plt.stackplot(flds, *lc_stats.values.reshape(n_cycles -
                                                 1, n_fields), labels=cycles)
    plt.legend(loc=2)


def get_fld_cell_counts(expr_DAPIs):
    """ get number of cells for each field.

    Returns
    --------
    fld_stat: pd.Series
        cell counts of each field in Series.
    """
    fld_stat = pd.DataFrame(index=expr_DAPIs.index, columns=['fld'])
    fld_stat.loc[:, 'fld'] = [
        '_'.join(x.split('_')[:-1]) for x in fld_stat.index]
    fld_stat = fld_stat.fld.value_counts()
    return fld_stat


def get_mean_lost_cell_fraction(expr_DAPIs, lc, fld_stat_method='overall'):
    """ Calculate mean fraction of lost cells.

    Parameters
    --------
    lc: list-like
        list of lost cells
    fld_stat_method: str
        method for calculating mean lost cell fraction. 'overall' will ignore 
        individual fields.

    Returns
    --------
    per_fld_v_fraction: pd.Series
        fraction of lost cells in each field.
    mean_valid_cell_fraction: numeric
        as named
    """
    if fld_stat_method == 'overall':
        mean_valid_cell_fraction = len(lc) / expr_DAPIs.shape[0]
        per_fld_v_fraction = None
        std_valid_cell_fraction = None,
    else:
        all_cells_fld_stat = get_fld_cell_counts(expr_DAPIs)
        valid_cells = expr_DAPIs.drop(lc)
        valid_cells_fld_stat = get_fld_cell_counts(valid_cells)
        per_fld_v_fraction = valid_cells_fld_stat.reindex(
            all_cells_fld_stat.index, fill_value=1) / all_cells_fld_stat
        per_fld_v_fraction = 1 - per_fld_v_fraction
        mean_valid_cell_fraction = per_fld_v_fraction.mean()
        std_valid_cell_fraction = per_fld_v_fraction.std()
    return per_fld_v_fraction, mean_valid_cell_fraction, std_valid_cell_fraction


def ROC_lostcells(expr_DAPIs, cutoff_min=1, cutoff_max=3,
                  steps=20, n_cycles=8, elbow_threshold=0.1,
                  filtering_method='bgnd',
                  fld_stat_method='individual',
                  segmentation_cycle=4):
    """ Plot curve of filtering threshold (x) and mean fraction of lost cells per field (y). 
    Optimal threshold are determined as the 'elbow point' of the curve. Filtering method can be 
    based on cycle variation or background. 


    Parameters
    --------
    expr_DAPIs: pd.DataFrame
        table of cells by channel/markers. First column need to be background, and the following
        columns will be aranged from cycle 1 to cycle n.
    cutoff_min(max): numeric
        minimal or maximal threshold for determining lost cells.
    steps: int
        total number of candidate threshold values within the previously defined range.
    n_cycles: int
        number of cycles in experiment.
    filtering_method: str
        method to determine lost cells. 'bgnd' or 'cycle_diff'. 'bgnd' for background based 
        thresholding and 'cycle_diff' for cycle variation based approach.
    fld_stat_method: str
        method for calculating mean lost cell fraction. 'overall' will ignore 
        individual fields.
    segmentation_cycle: int
        cycle where segmentation occurred. 
    elbow_threshold: numeric
        currently not implemented.

    Returns
    --------
    lost_cells: list
        list of lost cell indices.
    """
    x = np.linspace(cutoff_min, cutoff_max, steps)
    y = []

    if filtering_method == 'bgnd':
        if cutoff_max <= 100:
            x_title = 'Fold of background FI as cutoff'
        else:
            x_title = 'Additional absolute background FI as cutoff'

        for j in x:
            lost_cells = get_lost_cells(
                expr_DAPIs, j, n_cycles, filtering_method, segmentation_cycle)
            _, fraction_lost_cells, _ = get_mean_lost_cell_fraction(
                expr_DAPIs, lost_cells, fld_stat_method, )
            y.append(fraction_lost_cells)

    elif filtering_method == 'cycle_diff':
        x_title = 'Fold of between cycle FI variation as cutoff'
        for j in x:
            lost_cells = get_lost_cells(
                expr_DAPIs, j, n_cycles, filtering_method, segmentation_cycle)
            _, fraction_lost_cells, sd_fraction_lost_cells = get_mean_lost_cell_fraction(
                expr_DAPIs, lost_cells, fld_stat_method)
            y.append(fraction_lost_cells)

    # angle method to determine the elbow. adapted from here.
    # https://pdfs.semanticscholar.org/25d3/84f032b4d0d55019de354e32675d329f98df.pdf
    local_max = []
    for j in range(1, len(y) - 1):
        _local_max = y[j - 1] + y[j + 1] - 2 * y[j]
        local_max.append(_local_max)
    # ignores first elbow which reflect lost cells
    # the following elbow corresponds to moved cells.
    if filtering_method == 'cycle_diff':
        j_max = np.argmax(local_max)
        local_max[j_max] = np.min(local_max)
    local_max = local_max / np.ptp(local_max)
    elbow_idx = np.argmax(local_max > 0.25 * np.max(local_max)) + 1
    # Plotting
    plt.plot(x, y)
    plt.plot([x[elbow_idx], x[elbow_idx]], [0, 1],
             color='red', linestyle='dashed')
    text_offset = (cutoff_max - cutoff_min) / steps
    plt.text(x[elbow_idx] + 2 * text_offset, y[elbow_idx] - 0.05,
             'x = {:.2f}, y = {:.2f}'.format(x[elbow_idx], y[elbow_idx]))
    plt.xlabel(x_title)
    plt.ylabel('Fraction of valid cells')
    return x, y, x[elbow_idx]
