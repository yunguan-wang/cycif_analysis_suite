import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from fuzzywuzzy import process
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product


def pie_chart_from_well(well_fld_mean, ax=None, well_id=None, plot_legend=False, cmap=None, norm=None):
    """Basic function for Cycif QC. Makes a radar plot where each brach represents an column in the input data.
    Different columns are shown in different colors. Metadata must have field, plate and well information.

    Paramerters
    --------
    ax: None or matplotlib.Axes
        Current ax to plot on.
    well_id: None or str
        Name of the well to be used as figure title, if None, no title are drawn.
    plot_legend: bool
        Whether or not to plot legend, should be turned off if used in a wrapper.
    cmap: matplotlib.colors.Colormap
        cmap object for color mapping.
    norm: matplotlib.colors
        color range mapping.

    """
    if cmap is None:
        cmap = sns.light_palette("green", as_cmap=True)
    if isinstance(well_fld_mean, pd.DataFrame):
        well_fld_mean = well_fld_mean.iloc[:, 0]
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=15, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [mapper.to_rgba(x) for x in well_fld_mean.values]
    ax = ax or plt.gca()
    if ax is None:
        ax = plt.subplot(111, polar=True)
    well_fld_mean.fillna(0, inplace=True)
    ax.pie([1] * well_fld_mean.shape[0],
           colors=colors, labels=well_fld_mean.index)
    ax.set_title(well_id, loc='left', fontdict={'fontsize': 14}, pad=2)


def get_fld_mean(plate_meta, expr_data, well_col='Well', fld_col='fld'):
    """ Calculate per field mean of each field in each well. Assumes samples 
    are from the same plate.

    Parameters
    --------
    plate_meta: pandas.DataFrame
        a table of metadata, rows as cell names and columns as metadata.
    expr_data: pandas.DataFrame
        a table of expression data, rows as cell names and columns as 
        channel/markers.

    Returns
    --------
    fld_mean: pandas.DataFrame
        a table of per field per well mean intensity of each channel.
    """
    expr_data = expr_data.reindex(plate_meta.index).copy()
    expr_data[well_col] = plate_meta[well_col]
    expr_data[fld_col] = plate_meta[fld_col]
    fld_mean = expr_data.groupby([well_col, fld_col]).mean()
    return fld_mean


def process_channel_info(channel_info, expr_data):
    """ Static function for processing channel info. 
    """
    channel_info = channel_info[channel_info.Channel != 'DAPI'].copy()
    channel_info.sort_values(['Cycle', 'Channel'], inplace=True)
    channel_info.index = np.arange(channel_info.shape[0])
    choices = channel_info.Marker.values
    for col in expr_data.columns:
        if 'DNA' in col:
            marker = 'Hoechst'
            cycle = 1
        elif ' (alpha-isoform)' in col:
            marker = process.extract(col.replace(
                ' (alpha-isoform)', 'alpha'), choices, limit=1)[0][0]
        elif 'bckgrnd.1' in col:
            marker = 'cy3 bckgrnd'
        elif 'bckgrnd.2' in col:
            marker = 'cy5 bckgrnd'
        elif 'bckgrnd.3' in col:
            marker = 'FITC bckgrnd'
        else:
            marker = process.extract(col, choices, limit=1)[0][0]
        marker_idx = channel_info[channel_info.Marker == marker].index
        channel_info.loc[marker_idx, 'expr_col'] = col
    channel_info_Nuclei = channel_info.copy()
    channel_info_Nuclei.expr_col = channel_info_Nuclei.expr_col.apply(
        lambda x: x.replace('Cytoplasm', 'Nuclei'))
    channel_info = channel_info.append(channel_info_Nuclei)
    channel_info.index = np.arange(channel_info.shape[0])
    channel_info = channel_info[channel_info.Marker != 'empty']
    return channel_info


def plate_QC_pdf(plate_fld_mean, channel_info, figname):
    """ Generate a multipage pdf file for cells on each cycif plate. 
    Each page is organized in a 3 X 3 grid where rows represent cycles 
    and columns represent different channels. 

    Parameters
    --------
    plate_fld_mean: pandas.DataFrame
        a table of per well per field intensity mean of each channel/marker.
    channel_info: pandas.DataFrame
        a table of channel/marker metadata including cycle the marker is measured, 
        Must have columns named 'Cycle', 'Channel', 'Marker', and 'expr_col'. 
        This table should be already sorted by ['Cycle', 'Channel'] and indexed.
        column name of the marker in the 'plate_fld_mean' DataFrame, dye used. These
        combined determined the location on the pdf file of the particular marker.
    figname: str
        figure name to be used.

    """
    pdf = PdfPages(figname)
    marker_orders = channel_info.index.values
    k = 0
    wells = []
    for row, col in product(['C', 'D', 'E'], np.arange(4, 11)):
        wells.append(row + str(col).zfill(2))
    # splitting the channel/markers into pages. Each row in the `channel_info' dataframe
    # will be shown as a subplot, and thus for every 9th row, a new page will
    # be created.
    while k * 9 < marker_orders.max():
        # get data for the new page.
        new_page_channel_info = channel_info[
            (marker_orders < 9 * (k + 1)) & (marker_orders >= 9 * k)]
        # adjusting the index to fit in the 3x3 indexing system
        new_page_channel_info.index = [
            x - 9 * k for x in new_page_channel_info.index]
        # using a outer grid of 3x3 and inner grid of 3X7 for each page
        fig = plt.figure(figsize=(32, 16))
        outer = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)
        all_flds = ['fld' + str(x) for x in range(1, 10)]
        # iterate through each row. Again, each row will be represented as a
        # subplot.
        for i in range(9):
            try:
                col_name = new_page_channel_info.loc[i, 'expr_col']
            except KeyError:
                continue
            fld_mean_col = plate_fld_mean[col_name]
            cycle = new_page_channel_info.loc[i, 'Cycle']
            # setting colors
            if new_page_channel_info.loc[i, 'Channel'] == 'FITC':
                cmap = sns.light_palette("green", as_cmap=True)
            elif new_page_channel_info.loc[i, 'Channel'] == 'cy3':
                cmap = sns.light_palette("red", as_cmap=True)
            elif new_page_channel_info.loc[i, 'Channel'] == 'cy5':
                cmap = sns.light_palette("purple", as_cmap=True)
            # color values are normlized to the plate-wise max of each
            # chennel/marker.
            color_scale = matplotlib.colors.Normalize(
                vmin=0, vmax=fld_mean_col.max(), clip=True)

            # write figure annotations including channel name and cycles.
            # Channel names
            col_title_x = 0.24 + int(i % 3) * 0.27
            col_title_y = 0.9 - (cycle % 3 * 0.262)
            fig.text(col_title_x, col_title_y, col_name,
                     horizontalalignment='center', fontsize='x-large')
            # Cycles
            if i % 3 == 0:
                fig.text(0.12, 0.78 - cycle % 3 * 0.26, 'Cycle ' + str(cycle),
                         rotation='vertical', horizontalalignment='center', fontsize='x-large')

            # plotting inner grids. Organized by wells. Rows as well C, D, E and columns from
            # X04 to X10.
            inner = gridspec.GridSpecFromSubplotSpec(3, 7,
                                                     subplot_spec=outer[i],
                                                     wspace=0.02,
                                                     hspace=0.02)
            available_wells = plate_fld_mean.index.levels[0]
            for j in range(21):
                # subsetting data for the current well to be plotted.
                current_well = wells[j]
                if current_well not in available_wells:
                    continue
                fld_mean_well = fld_mean_col.loc[current_well]
                missing_flds = [
                    x for x in all_flds if x not in fld_mean_well.index]
                for missing_fld in missing_flds:
                    fld_mean_well[missing_fld] = 0
                fld_mean_well.index = [
                    x.replace('fld', '') for x in fld_mean_well.index]
                fld_mean_well.sort_index(inplace=True)
                # Add subplot using the pie_chart_from_well.
                ax = plt.Subplot(fig, inner[j])
                pie_chart_from_well(
                    fld_mean_well, ax=ax, cmap=cmap, norm=color_scale, well_id=current_well)
                fig.add_subplot(ax)
        pdf.savefig(fig)
        plt.close()
        k += 1
    pdf.close()

"""
=== Functions below are not used currently.

"""


def bar_chart_from_well(fld_mean, ax=None, well_id=None, plot_legend=False, cmap=None):
    """Basic function for Cycif QC. Makes a radar plot where each brach represents an column in the input data.
    Different columns are shown in different colors. Metadata must have field, plate and well information.

    Paramerters
    --------
    ax: None or matplotlib.Axes
        Current ax to plot on.
    well_id: None or str
        Name of the well to be used as figure title, if None, no title are drawn.
    plot_legend: bool
        Whether or not to plot legend, should be turned off if used in a wrapper.

    """
    fld_mean = fld_mean.iloc[:, 0]
    if cmap is None:
        cmap = sns.color_palette("Set1", n_colors=9)
        cmap = pd.Series(
            cmap, index=['fld' + str(x) for x in range(1, 10)])
    ax = ax or plt.gca()
    if ax is None:
        ax = plt.subplot(111, polar=True)
    fld_mean.fillna(0, inplace=True)
    for idx in fld_mean.index:
        ax.bar(idx, fld_mean[idx], color=cmap[idx], label=idx)
    ax.set_title(well_id, loc='left', fontdict={'fontsize': 14}, pad=10)


def QC_plot_all(metadata, expr, combine_output_fn=None, plate_col='Plate', well_col='Well', replicate_col=None, **kwargs):
    """Overall wrapper function for experiment-wise QC. Makes field-wise of mean intensity plot for each well in each plate.
    Metadata must have field, plate and well information.

    Paramerters
    --------
    metadata: pandas.DataFrame
        Table of Cycif data metadata. Must have matching indicies to expr.
    expr: pandas.DataFrame or pandas.Series.
        Table of cycif expression data. Rows as cells and columns as channels. Best log normalized. 
        Can be used to visualize a single channel if expr is passed as a Series.
    combine_output_fn: None or str
        Name of pdf file that all figures are combined into. If let None, each plate will be represented as one figure.
    """
    if combine_output_fn is not None:
        pdf = PdfPages(combine_output_fn)

    for plate in metadata.groupby(plate_col):
        plate_id, plate_meta = plate
        plate_meta = plate_meta.sort_values(well_col)
        if combine_output_fn is not None:
            fig_title = ' '.join([expr.name, 'plate', str(plate_id)])
            fig = QC_plot_from_plate(
                plate_meta, expr, well_col=well_col, fig_title=fig_title, **kwargs)
            pdf.savefig(fig)
            plt.close()

        else:
            if replicate_col is None:
                figname = 'Plate {}.png'.format(plate_id)
            else:
                figname = 'Replicate {} Plate {}.png'.format(
                    plate_meta[replicate_col][0], plate_id)
            QC_plot_from_plate(plate_meta, expr, fig_name=figname,
                               well_col=well_col, fig_title=None, **kwargs)
    if combine_output_fn is not None:
        pdf.close()


def QC_plot_from_plate(plate_meta, expr, plot_fun='pie',
                       fig_name=None, well_col='Well', field_col='fld', size=5, fig_title=None):
    """Wrapper function for plate-wise QC. Makes field-wise of mean intensity plot for each well.
    Metadata must have field, plate and well information.

    Paramerters
    --------
    plate_meta: pandas.DataFrame
        Table of Cycif data metadata of a ceritain plate. Must have matching indicies to expr.
    expr: pandas.DataFrame or pandas.Series.
        Table of cycif expression data. Rows as cells and columns as channels. Best log normalized. 
        Can be used to visualize a single channel if expr is passed as a Series.
    (well,field)_col: str
        Column names for well and field metadata in 'plate_meta'
    fig_name: None or str
        Name of output file, if None, show figure in console.
    size: numeric
        Size factor of each subplots.

    Returns
    --------
    fig: matplotlib.figure
        Figure object to be used in pdf outputs.
    """
    # define plotting functions.
    subplot_kw = None
    if plot_fun == 'pie':
        plot_fun = pie_chart_from_well

    elif plot_fun == 'rader':
        plot_fun = radar_plot_from_well
        subplot_kw = dict(polar=True)
    elif plot_fun == 'bar':
        plot_fun = bar_chart_from_well
    plot_cmap = 'Blues'
    expr = pd.DataFrame(expr)
    expr_cols = expr.columns
    plate_meta = plate_meta.merge(
        expr, left_index=True, right_index=True, how='left', sort=False)
    plate_meta = plate_meta.sort_values([well_col, field_col])
    num_wells = plate_meta[well_col].unique().shape[0]
    ncols = 7
    nrows = int(np.ceil(num_wells / ncols))
    plate_df_fld_mean = plate_meta.groupby(['Well', 'fld'], sort=False).mean()
    plate_df_fld_mean.reset_index(inplace=True)
    fld_mean_max = plate_df_fld_mean.iloc[:, -1].max()
    color_scale = matplotlib.colors.Normalize(
        vmin=0, vmax=fld_mean_max, clip=True)
    # get all field names.
    all_unique_fileds = sorted(plate_df_fld_mean[field_col].unique())
    fig, axes = plt.subplots(nrows, ncols,
                             subplot_kw=subplot_kw,
                             figsize=(ncols * 4, nrows * 4),
                             sharex=True,
                             sharey=True)
    axes = axes.ravel()
    idx = 0
    for well in plate_df_fld_mean.groupby(well_col):
        well_id, well_df = well
        fld_mean = pd.DataFrame(0, index=all_unique_fileds, columns=expr_cols)
        fld_mean.loc[well_df[field_col],
                     expr_cols] = well_df[expr_cols].values
        fld_mean.name = well_id
        # Adding back missing fields
        missing_flds = [
            x for x in all_unique_fileds if x not in fld_mean.index]
        for missing_fld in missing_flds:
            fld_mean[missing_fld] = 0
        fld_mean.sort_index(inplace=True)
        pie_chart_from_well(fld_mean, ax=axes[idx], well_id=well_id,
                            norm=color_scale, cmap=plot_cmap)
        idx += 1
    if fig_title is None:
        fig_title = ''
    fig.suptitle(fig_title, fontsize=18)
    # Make legend or color bar, if pie chart is used.
    if plot_fun == 'rader':
        plt.legend(loc=1, bbox_to_anchor=(0.5, -0.2),
                   fontsize='x-large', ncol=min(5, len(expr_cols)))
    else:
        temp_ax = fig.add_axes([.9, .25, .05, .5])
        cb_ax = fig.add_axes([.92, .3, .02, .33])
        g = temp_ax.imshow(np.arange(16).reshape(
            4, 4), cmap=plot_cmap, norm=color_scale)
        cb = plt.colorbar(g, cax=cb_ax, orientation='vertical')
        temp_ax.remove()
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        return axes


def radar_plot_from_well(input_df, ax=None, well_id=None, plot_legend=False):
    """Basic function for Cycif QC. Makes a radar plot where each brach represents an column in the input data.
    Different columns are shown in different colors. Metadata must have field, plate and well information.

    Paramerters
    --------
    ax: None or matplotlib.Axes
        Current ax to plot on.
    well_id: None or str
        Name of the well to be used as figure title, if None, no title are drawn.
    plot_legend: bool
        Whether or not to plot legend, should be turned off if used in a wrapper.

    returns
    --------
    ax: matplotlib.Axes
        Subplot object of radar plot.
    """
    ax = ax or plt.gca()
    if ax is None:
        ax = plt.subplot(111, polar=True)   # Set polar axis
    input_df = pd.DataFrame(input_df)
    for col in input_df.columns:
        series_data = input_df[col]
        labels = series_data.index
        values = series_data.values
        angles = np.linspace(0, 2 * np.pi, len(labels),
                             endpoint=False)  # Set the angle
        # close the plot
        values = np.concatenate((values, [values[0]]))  # Closed
        angles = np.concatenate((angles, [angles[0]]))  # Closed
        # Draw the plot (or the frame on the radar chart)
        ax.plot(angles, values, 'o-', linewidth=2, label=col)
        ax.fill(angles, values, alpha=0.25)  # Fulfill the area
    # Set the label for each axis
    ax.set_thetagrids(np.linspace(
        0, 360, len(labels), endpoint=False), labels)
    # ax.set_rlim(0,250)
    ax.grid(True)
    ax.tick_params(pad=5, labelsize='large')
    ax.set_title(well_id, loc='left', fontdict={'fontsize': 14}, pad=10)
    if plot_legend:
        plt.legend()
    return ax
