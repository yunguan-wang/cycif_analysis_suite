import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycifsuite.common_apis import check_numeric


def get_lost_cells(expr_DAPIs, threshold, n_cycles,
                   filtering_method='bgnd',
                   direction='down',
                   segmentation_cycle=4):
    """find lost cells based on one of the two algorithms. If background based
    approach is used, cells with DAPI signal at segmentation cycle less than background 
    plus a threshold will be dropped. The threshold is inferred as fold if it is less 
    than 100, otherwise it is interpreted as abosolute value to be added to background.
    if cycle variation method is used, DAPI signal of each cycle will be divided by the 
    difference betweeen signal of current cycle to that of the next one. The quotient will
    be compared with the threshold. 

    Parameters
    --------
    expr_DAPIs : pd.DataFrame
        table of cells by channel/markers. First column need to be background, and the following
        columns will be aranged from cycle 1 to cycle n. Data should NOT be log normalized.
    threshold : numeric
        threshold for determining lost cells.
    n_cycles : int
        number of cycles in experiment.
    filtering_method : str
        method to determine lost cells. 'bgnd' or 'cycle_diff'. 'bgnd' for background based 
        thresholding and 'cycle_diff' for cycle variation based approach.
    segmentation_cycle : int
        cycle where segmentation occurred. 

    Returns
    --------
    lost_cells : pandas.Series
        table lost cell indices with the cycle where it is lost.
    """
    all_cells = expr_DAPIs.index
    lost_cells = pd.Series(index=all_cells)

    if filtering_method == 'bgnd':
        bgnd = expr_DAPIs.iloc[:, 0]
        threshold = threshold * bgnd
        current_cycle_fi = expr_DAPIs.iloc[:, segmentation_cycle]
        current_lost_cells = all_cells[(current_cycle_fi <= threshold)]
        lost_cells[current_lost_cells] = 'lost_due_to_high_bgnd'

    elif filtering_method == 'cycle_diff':
        for i in range(n_cycles - 1, 0, -1):
            current_cycle_fi = expr_DAPIs.iloc[:, i]
            previous_cycle_fi = expr_DAPIs.iloc[:, i - 1]
            # cycle_diff = abs(previous_cycle_fi - current_cycle_fi)
            if direction == 'down':
                # cycle_diff = previous_cycle_fi - current_cycle_fi
                current_lost_cells = all_cells[
                    (current_cycle_fi < threshold * previous_cycle_fi)]
            else:
                if i > 3:
                    continue
                current_lost_cells = all_cells[
                    (previous_cycle_fi < threshold * current_cycle_fi)]
                # current_lost_cells = all_cells[
                #     (current_cycle_fi > previous_cycle_fi / threshold)]
            lost_cells[current_lost_cells] = 'Cycle_' + str(i + 1)
    lost_cells = lost_cells.dropna()
    return lost_cells, lost_cells.index.tolist()


def get_fld_cell_counts(expr_DAPIs):
    """get number of cells for each field.

    Returns
    --------
    fld_stat : pd.Series
        cell counts of each field in Series.
    """
    fld_stat = pd.DataFrame(index=expr_DAPIs.index, columns=['fld'])
    fld_stat.loc[:, 'fld'] = [
        '_'.join(x.split('_')[:-1]) for x in fld_stat.index]
    fld_stat = fld_stat.fld.value_counts()
    return fld_stat


def get_mean_lost_cell_fraction(expr_DAPIs, lc, fld_stat_method='overall'):
    """Calculate mean fraction of lost cells.

    Parameters
    --------
    lc : list-like
        list of lost cells
    fld_stat_method : str
        method for calculating mean lost cell fraction. 'overall' will ignore 
        individual fields.

    Returns
    --------
    per_fld_v_fraction : pd.Series
        fraction of lost cells in each field.
    mean_valid_cell_fraction : numeric
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


def find_elbow(x, y, left_stepping=False):
    """Improved algorithm with modified angle method. Basically, 
    it is looking for maximal angle formed by three consecutive points.
    angle method to determine the elbow. adapted from here.
    https://pdfs.semanticscholar.org/25d3/84f032b4d0d55019de354e32675d329f98df.pdf

    Parameters
    --------
    x : list
        list a threshoulds.
    y : list
        list of fractions of lost cells.
    left_stepping : bool
        Experimental feature. This is used to 'step' back from the peak in the angle
        plot to the first peak to the left to allow for a slightly more lenient threshold. 


    Returns
    --------
    elbow_idx : int
        index of elbow on the threshold vector.
    """

    angle = []
    x_step = x[1] - x[0]
    for x_idx in range(1, len(y) - 1):
        angle_left = 180 / np.pi * \
            np.arctan(x_step / (y[x_idx] - y[x_idx - 1]))
        angle_right = 180 / np.pi * \
            np.arctan(x_step / (y[x_idx + 1] - y[x_idx]))
    #     print(angle_left, angle_right)
        local_angle = 180 + angle_left - angle_right
        angle.append(local_angle)
    elbow_idx = np.argmax(angle) + 1

    if left_stepping:
        print('Use the first peak left from the highest angle instead...')
        # starting from the first angle left from the max angle
        for angle_idx in range(elbow_idx - 2, 1, -1):
            peak = angle[angle_idx - 1] + \
                angle[angle_idx + 1] - 2 * angle[angle_idx]
            if peak < 0:
                break
        elbow_idx = angle_idx + 1

    # older algorithm
    # local_max = []
    # for j in range(1, len(y) - 1):
    #     _local_max = y[j - 1] + y[j + 1] - 2 * y[j]
    #     local_max.append(_local_max)
    # # ignores first elbow which reflect lost cells
    # # the following elbow corresponds to moved cells.
    # local_max = local_max / np.ptp(local_max)
    # # elbow_idx = np.argmax(local_max > 0.25 * np.max(local_max)) + 1
    # elbow_idx = np.argmax(local_max) + 1

    return elbow_idx, angle


def plot_elbow(x, y, angle, elbow_idx, color='red', figname=None):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 6))
    axes = axes.ravel()
    threshold_plot = axes[0]
    angle_plot = axes[1]
    # plotting threshold by fraction of lost cells
    threshold_plot.plot(x, y)
    threshold_plot.plot([x[elbow_idx], x[elbow_idx]], [
        0, 1], color=color, linestyle='dashed')
    threshold_plot.set_title('x = {:.2f}'.format(x[elbow_idx]))
    threshold_plot.set_xlabel('Threshold', fontsize=16)
    threshold_plot.set_ylabel('Fraction of lost cells', fontsize=16)
    # plotting angles by thresholds.
    angle_plot.plot(x[1:-1], angle)
    angle_plot.plot([x[elbow_idx], x[elbow_idx]], [min(angle), max(angle)],
                    color=color, linestyle='dashed')
    angle_plot.set_ylabel(
        'Angles formed by 3 consecutive points', fontsize=16)
    # save plots
    if figname is not None:
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()


def ROC_lostcells(expr_DAPIs, cutoff_min=1, cutoff_max=3,
                  steps=20, n_cycles=8, elbow_threshold=0.1,
                  filtering_method='bgnd',
                  fld_stat_method='individual',
                  figname=None,
                  automatic=True,
                  left_stepping=False,
                  **kwargs):
    """Plot curve of filtering threshold (x) and mean fraction of lost cells per field (y). 
    Optimal threshold are determined as the 'elbow point' of the curve. Filtering method can be 
    based on cycle variation or background. 


    Parameters
    --------
    expr_DAPIs : pd.DataFrame
        table of cells by channel/markers. First column need to be background, and the following
        columns will be aranged from cycle 1 to cycle n.
    cutoff_min(max) : numeric
        minimal or maximal threshold for determining lost cells.
    steps : int
        total number of candidate threshold values within the previously defined range.
    n_cycles : int
        number of cycles in experiment.
    filtering_method : str
        method to determine lost cells. 'bgnd' or 'cycle_diff'. 'bgnd' for background based 
        thresholding and 'cycle_diff' for cycle variation based approach.
    fld_stat_method : str
        method for calculating mean lost cell fraction. 'overall' will ignore 
        individual fields.
    segmentation_cycle : int
        cycle where segmentation occurred. 
    elbow_threshold : numeric
        currently not implemented.
    automatic : bool
        find threshold using suggested method, else use interactive approach to assist selection of 
        threshold.
    left_stepping : bool
        whether to use left stepping in elbow detection.

    Returns
    --------
    lost_cells : list
        list of lost cell indices.
    """
    x = np.linspace(cutoff_min, cutoff_max, steps)
    y = []

    if filtering_method == 'bgnd':
        for j in x:
            _, lost_cells = get_lost_cells(
                expr_DAPIs, j, n_cycles, filtering_method, **kwargs)
            _, fraction_lost_cells, _ = get_mean_lost_cell_fraction(
                expr_DAPIs, lost_cells, fld_stat_method, )
            y.append(fraction_lost_cells)

    elif filtering_method == 'cycle_diff':
        for j in x:
            _, lost_cells = get_lost_cells(
                expr_DAPIs, j, n_cycles, filtering_method, **kwargs)
            _, fraction_lost_cells, _ = get_mean_lost_cell_fraction(
                expr_DAPIs, lost_cells, fld_stat_method)
            y.append(fraction_lost_cells)

    # angle method to determine the elbow. adapted from here.
    # https://pdfs.semanticscholar.org/25d3/84f032b4d0d55019de354e32675d329f98df.pdf
    elbow_idx, angle = find_elbow(x, y, left_stepping)
    # Plotting
    plot_elbow(x, y, angle, elbow_idx, figname=figname)
    # give the option to have some user adjustment for the threshold.
    if not automatic:
        while True:
            arg = input(
                'Suggested threshold value at {}, use suggested (y), do left_stepping (l) or enter your own ([float])'.format(x[elbow_idx]))
            if arg == 'y':
                print('Use suggested value...')
                elbow_x = x[elbow_idx]
                break
            elif arg == 'l':
                elbow_idx, _ = find_elbow(x, y, left_stepping=True)
                elbow_x = x[elbow_idx]
                break
            elif isinstance(check_numeric(arg), float):
                elbow_x = check_numeric(arg)
                print('Use manual input elbow at {:.2f}'.format(elbow_x))
                break
        # new threshold plot
        plot_elbow(x, y, angle, elbow_idx, 'green', figname=figname)
    else:
        elbow_x = x[elbow_idx]
    return x, y, elbow_x


def find_bad_regions(lc_fraction, bad_field_cutoff=0.9, automatic=False):
    """Find regions such as fields, well or rows on a plate where it fails 
    significantly more often than others. Cut off are hard coded as mean + 4SD.
    This is designed to be used in plate based per field experiments.

    Parameters
    --------
    lc_fraction : pandas.DataFrame
        a table of per field lost cell ratio, which is the first columns.
        Additional columns are the regions to be tested, could be a well,
        a row or even a row on a plate.
    bad_field_cutoff : float
        a cutoff threshold meaning the max acceptable ratio of lost cells in a field. 
        If a field has more cells lost than the cutoff, the whold field it will be
        considered as a bad field.
    automatic : bool
        if True, no prompt will be printed out asking for human decision. 
        Not recommended for unfamiliar data. Default is False.

    Returns
    --------
    bad_idx : list
        a list of indices of the input lc_fraction table considered as bad fields.
    """
    t = lc_fraction.iloc[:, 0]
    bad_idx = t[t >= bad_field_cutoff].index.tolist()
    additional_bad_idx = []
    for col in lc_fraction.columns[1:]:
        # get bad condition ratio for the metadata defined by the colname.
        col_crit = lc_fraction.groupby(col).agg(
            lambda x: (x > bad_field_cutoff).sum() / x.shape[0])
        col_crit = pd.DataFrame(col_crit)
        # evaluate each condition based on other's mean and std.
        col_crit['others_mean'] = col_crit.apply(
            lambda x: col_crit.drop(x.name).iloc[:, 0].mean(), axis=1)
        col_crit['others_std'] = col_crit.apply(
            lambda x: col_crit.drop(x.name).iloc[:, 0].std(), axis=1)
        col_crit['crit'] = col_crit.iloc[:, 0] > (
            col_crit.others_mean + 4 * col_crit.others_std)
        for bad_cond in col_crit.index[col_crit.crit]:
            bad_cond_value = col_crit.loc[bad_cond][0]
            print('{} with a bad ratio at {:.2f} is considered a bad {}'.format(
                bad_cond, bad_cond_value, col))
            if automatic:
                to_drop = 'y'
            else:
                to_drop = input('Drop {}? (y/n)'.format(bad_cond))
            if to_drop == 'y':
                additional_bad_idx += lc_fraction[
                    lc_fraction[col] == bad_cond].index.tolist()
    bad_idx = list(set(bad_idx + additional_bad_idx))
    return bad_idx


def update_metadata(metadata, lost_cells, bad_pwfs=None):
    """Update metadata table with a new column indicating if a cell is considered as lost.
    Considers both direct lost cell list and an additional bad regions defined by 'find_bad_regions'.

    Parameters
    --------
    metadata : pandas.DataFrame
        table of metadata
    lost_cells : list
        list of indices of lost cells, must match those in metadata.
    bad_pwfs : None or list
        list of additional bad regions in the experiment. 

    Returns
    --------
    updated_metadata : pandas.DataFrame
        updated metadata with a new column 'labeled_as_lost' indicating if a cell is lost.
    """
    updated_metadata = metadata.copy()
    updated_metadata['pwf'] = [
        '_'.join(x.split('_')[:-1]) for x in updated_metadata.index]
    updated_metadata['labeled_as_lost'] = 'No'
    updated_metadata.loc[lost_cells, 'labeled_as_lost'] = 'Yes'
    if bad_pwfs is not None:
        updated_metadata.loc[updated_metadata.pwf.isin(
            bad_pwfs), 'labeled_as_lost'] = 'Yes'
    return updated_metadata


def plot_lost_cell_per_cycle_stacked_area(lc_cycle, metadata=None, in_fraction=False,
                                          n_fields=9, n_cycles=8, figname=None, return_lc_stats=False):
    """Plot stacked plot of lost cells per field per cycle. Can make plot based on absolute count of lost cells
    or based on average fraction of lost cells per well per cycle. If the letter, metadata of cells must be 
    provided.

    Parameters
    --------
    lc_cycle : pd.Series
        a table of each lost cell and the cycle it was lost.
    metadata : pd.DataFrame
        a table of cell metadata.
    in_fraction : bool
        whether to calculate fraction of lost cells.
    n_fields : int
        number of fields in the experiment.
    n_cycles : int
        number of cycles in the expriment.
    figname : str or None
        if None, plot to console, otherwise save as file.
    return_lc_stats : bool
        whether to return processed lc_stats DataFrame.

    Returns
    --------
    lc_stats : 
        the table used in the final plotting.

    """
    lc_cycle = pd.DataFrame(lc_cycle, columns=['cycle'])
    lc_cycle['field'] = [x.split('_')[-2] for x in lc_cycle.index]
    lc_cycle['count'] = 1
    lc_cycle['pw'] = ['_'.join(x.split('_')[:-1]) for x in lc_cycle.index]
    if not in_fraction:
        lc_stats = lc_cycle.groupby(['cycle', 'field']).sum()
        cycles = lc_stats.index.levels[0]
        flds = lc_stats.index.levels[1]
        # for occasions where a field is missing from a cycle.
        idx = pd.MultiIndex.from_product((cycles, flds))
        lc_stats = lc_stats.reindex(idx, fill_value=0)
        plot_data = lc_stats
    else:
        lc_stats = lc_cycle.groupby(['cycle', 'field', 'pw'])['count'].sum()
        metadata['field'] = [x.split('_')[-2] for x in metadata.index]
        metadata['pw'] = ['_'.join(x.split('_')[:-1]) for x in metadata.index]
        metadata['count'] = 1
        all_cell_stats = metadata.groupby(['field', 'pw'])['count'].sum()
        flds = metadata.field.unique()
        all_pws = metadata.pw.unique()
        multi_idx = all_cell_stats.index
        # plotting
        plot_data = pd.DataFrame()
        for cycle in lc_stats.index.levels[0]:
            current_cycle = lc_stats.loc[cycle]
            current_cycle = current_cycle.reindex(multi_idx, fill_value=0)
            current_cycle = current_cycle / all_cell_stats
            current_cycle = current_cycle.reset_index()
            current_cycle['cycle'] = cycle
            plot_data = plot_data.append(current_cycle)
        plot_data = plot_data.sort_values('field')
        cycles = sorted(plot_data.cycle.unique())
        plot_data = plot_data.groupby(['cycle', 'field']).mean()
    # plotting
    plt.stackplot(flds, *plot_data.values.reshape(n_cycles -
                                                  1, n_fields), labels=cycles)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center')
    plt.xlabel('Field', fontsize=18)
    plt.ylabel('Absolute accumulated cell loss', fontsize=18)
    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
    if return_lc_stats:
        return lc_stats


def plot_lc_by_fld(t):
    """Plot lost cells for each firld

    Parameters
    --------
    t : pd.Series
        a talbe returned by get_mean_lost_cell_fraction, with index of each field name and values as
        the fraction of lost cells.

    """
    sns.set(font_scale=2)
    t.name = 'Fraction_of_lost_cells'
    t = pd.DataFrame(t)
    t['field'] = [x.split('_')[-1] for x in t.index]
    t = t.sort_values('field')
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='field', y='Fraction_of_lost_cells', hue='field',
                data=t, dodge=False)
    plt.ylabel('Fraction of lost cells per field', fontsize=18)
    plt.legend(bbox_to_anchor=(1.1, 0.5), loc='center')
