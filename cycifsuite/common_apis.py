import pandas as pd
import numpy as np
import os
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp
from statsmodels.stats.multitest import fdrcorrection as fdr
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch


def channel_histograms(df, xlim=None, n_bins=100, save_fig_filename=None):
    """Make histograms for each column.

    Parameters
    --------
    df : pandas.DataFrame
        a table of cells by channel names. 
    """
    sns.set(font_scale=3)
    plt.rcParams['patch.linewidth'] = 0
    plt.rcParams['patch.edgecolor'] = 'none'
    n_rows = int(np.ceil(df.columns.shape[0] / 4))
    _, axes = plt.subplots(n_rows, 4, sharex=True)
    if xlim is None:
        pass
    else:
        plt.xlim(xlim)
    axes = axes.ravel()
    for i, col in enumerate(df.columns):
        figsize = (4 * 4, n_rows * 3)
        df[col].hist(bins=n_bins, ax=axes[i],
                     figsize=figsize, label=col, grid=False)
        axes[i].set_title(col, x=0.5, y=0.7)
    if save_fig_filename is None:
        plt.show()
    else:
        plt.savefig(save_fig_filename)
    plt.close()
    sns.set(font_scale=1)


def plot_clustering(df_low_dim, df_labels, figname=None):
    """Make scatter plots of dimention reduced data with cluster designation.

    Parameters
    --------
    df_low_dim : pandas.DataFrame or numpy ndarray.
        a table of low dimensional data, sample/cell by dimension. Must have at 
        least two dimensions.
    df_labels : pandas.Series or list-like.
        a table of sample/cell cluster designation. samples must be the same order
        as those in df_low_dim.
    """
    if isinstance(df_low_dim, pd.DataFrame):
        df_low_dim = df_low_dim.values
    if not isinstance(df_labels, pd.Series):
        df_labels = pd.Series(df_labels)
    unique_clusters = sorted(df_labels.unique())
    for cluster in unique_clusters:
        cells_idx = df_labels[df_labels == cluster].index.values
        plt.scatter(df_low_dim[cells_idx, 0], df_low_dim[
                    cells_idx, 1], label=cluster, s=1)

    plt.legend(markerscale=5, bbox_to_anchor=(1, 0.9))
    if figname is None:
        plt.savefig('Clustering_on_2D.png', bbox_inches='tight')
    else:
        plt.savefig(figname, bbox_inches='tight')
    plt.close()


def differential_analysis(test_samples, control_samples):
    """calculated fold change on log transformed data.

    Paremeters :
    --------
    test_samples : pandas.dataframe
        a dataframe of test data, rows as cells and columns as features.
    control_samples : pandas.dataframe
        a dataframe of control data, rows as cells and columns as features.
    """
    report = pd.DataFrame(
        columns=['logFC', 'T_pValue', 'KS_pValue', 'adj_T_pVal', 'adj_KS_pVal'])
    for feature in control_samples.columns:
        fc = test_samples[feature].mean() - control_samples[feature].mean()
        pval = ttest_ind(control_samples[feature], test_samples[feature])
        ks_pval = ks_2samp(control_samples[feature], test_samples[feature])[1]
        report.loc[feature, 'logFC'] = np.round(fc, 2)
        report.loc[feature, 'T_pValue'] = pval[1]
        report.loc[feature, 'KS_pValue'] = ks_pval
    report['adj_T_pVal'] = fdr(report.T_pValue)[1]
    report['adj_KS_pVal'] = fdr(report.KS_pValue)[1]
    return report


def plot_expr_on_2D(df_2d, df_raw_expr, figname, labels):
    """
    Make a grid of scatter plots of original data projected to a 2D space determined by UMAP.
    Each marker is visualized in a subplot and each point in the subplot is colored based on its original expression. 
    """
    plt.ioff()
    nrows = int(np.ceil(1 + len(df_raw_expr.columns) / 5))
    fig, ax = plt.subplots(
        nrows, 5, sharex='col', sharey='row', squeeze=False, figsize=(25, 5 * nrows))
    ax = ax.ravel()

    # plot control vs test
    labels.index = list(range(len(labels)))
    for label in labels.unique():
        idx = labels[labels == label].index
        ax[0].scatter(df_2d[idx, 0], df_2d[idx, 1],
                      cmap='bwr', s=1, label=label)
    ax[0].legend(markerscale=5, frameon=False)

    # plot markers
    idx = 1
    for colname in df_raw_expr.columns:
        subplot = ax[idx].scatter(df_2d[:, 0], df_2d[:, 1], c=df_raw_expr[
                                  colname].values, s=1, cmap='bwr', label=colname)
        ax[idx].legend(markerscale=5, frameon=False)
        fig.colorbar(subplot, ax=[ax[idx], ax[idx]],
                     pad=-0.05, extend='both', format='%.1f')
        idx += 1

    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def make_fc_time_plots(df_fc, control_name='DMSO'):
    sns.set(font_scale=2)
    time_order = sorted(df_fc.Time.unique())
    for drug in df_fc.DrugName.unique():
        if drug == control_name:
            continue
        sub_df_fc = df_fc[df_fc.DrugName == drug]
        sub_df_fc.reset_index(inplace=True)
        sub_df_fc = sub_df_fc.sort_values(['Dose', 'index'])
        sub_df_fc.Dose = np.log10(sub_df_fc.Dose)
        g = sns.FacetGrid(sub_df_fc, col_wrap=10, col='index',
                          hue='Time', hue_order=time_order, height=5)
        g = (g.map(plt.plot, 'Dose', 'logFC', marker=".", linewidth=5)
             .set_axis_labels("log10 Dose", "Dose")
             .set_titles(drug + " {col_name} ", weight='bold')
             .fig.subplots_adjust(wspace=.05, hspace=.1))
        plt.legend(bbox_to_anchor=(1.1, 0.6))
        plt.savefig(drug + ' marker logFC over time.png')
        plt.close()
    sns.set(font_scale=1)


def two_way_hc_ordering(data):
    lm = sch.linkage(data, method='complete')
    row_order = sch.dendrogram(lm, no_plot=True)['leaves']
    lm_t = sch.linkage(data.transpose(), method='complete')
    col_order = sch.dendrogram(
        lm_t, no_plot=True, distance_sort='ascending')['leaves']
    return row_order, col_order


def make_heatmap(df_data, figsize=(30, 9), vmax=4, vmin=-4,
                 fig_name=None, fig_title=None, cmap=None):

    sns.set(font_scale=1.5)
    plt.figure(figsize=(figsize))
    if cmap is not None:
        sns.heatmap(df_data, cmap=cmap, vmax=vmax, vmin=vmin, robust=True)
    else:
        sns.heatmap(df_data, vmax=vmax, vmin=vmin, robust=True)
    if fig_title is not None:
        plt.title(fig_title)
    plt.xticks(rotation=75)
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()
    sns.set(font_scale=1)


def calculate_cluster_composition(df_metadata):
    df_metadata['condition'] = df_metadata.DrugName + \
        '_' + df_metadata.Conc.astype(str)
    df_freq = df_metadata.groupby(['cluster', 'condition']).count().iloc[:, 0]
    df_freq = df_freq.groupby('cluster').apply(
        lambda x: x / x.sum()).reset_index()
    df_freq = df_freq.pivot(
        index='cluster', columns='condition', values='well')
    df_freq.fillna(0, inplace=True)
    rows, cols = two_way_hc_ordering(df_freq)
    df_freq = df_freq.iloc[rows, cols]
    drugs = [x.split('_')[0] for x in df_freq.columns]
    doses = [float(x.split('_')[1]) for x in df_freq.columns]
    df_freq = df_freq.transpose()
    df_freq.index = drugs
    df_freq.insert(0, 'Dose', doses)
    return df_freq.transpose()


"""
# Old function
def plot_dist_violinplot(df_dist,
    x_axis_col = 'Dose', 
    save_fig_filename = 'Drug_DMSO distance over dose and time.png') :

    sns.set(font_scale=1.5)
    new_df_dist = pd.DataFrame()
    if x_axis_col == 'Dose' :
        for time in df_dist.Time.unique() :
            dmso = df_dist[(df_dist.Time==time)&(df_dist.Drug==control_name)]
            low_dmso = dmso['Average Distance to DMSO'].quantile(0.01)
            high_dmso = dmso['Average Distance to DMSO'].quantile(0.99)
            dmso = dmso[dmso['Average Distance to DMSO'].between(low_dmso,high_dmso)]
            for drug in df_dist.Drug.unique() :
                if drug == control_name :
                    continue
                sub_df_dist = df_dist[(df_dist.Time==time)&(df_dist.Drug==drug)]   
                low = sub_df_dist['Average Distance to DMSO'].quantile(0.01)
                high = sub_df_dist['Average Distance to DMSO'].quantile(0.99)
                sub_df_dist = sub_df_dist[sub_df_dist['Average Distance to DMSO'].between(low,high)]
                dmso['Drug'] = drug
                sub_df_dist = sub_df_dist.append(dmso)
                new_df_dist = new_df_dist.append(sub_df_dist)
        df_dist = new_df_dist
        
    if x_axis_col == 'Dose' :    
        order = ['{ :.1e}'.format(x) for x in sorted(df_dist[x_axis_col].unique().astype(float))]
        g = sns.FacetGrid(data=df_dist,col='Drug',row = 'Time',palette='Blues',sharey=True, sharex=False,height=6)
        g = g.map(sns.violinplot,x_axis_col,'Average Distance to DMSO', order = order, bw=.2)
    elif x_axis_col == 'Time' :
        order = sorted(df_dist[x_axis_col].unique())
        g = sns.FacetGrid(data=df_dist,col='Drug',palette='Blues',sharey=True, sharex=True,height=6)
        g = g.map(sns.violinplot,x_axis_col,'Average Distance to DMSO', order = order, bw=.2)
    for ax in g.axes.flat :
        for label in ax.get_xticklabels() :
            label.set_rotation(60)
    plt.tight_layout()
    sns.set(font_scale=1)
    plt.savefig(save_fig_filename)
    plt.close()
    sns.set(font_scale=1)

# def make_fc_heatmap() :
#     # make heatmap
#     df_fc['condition'] = df_fc.DrugName + '_' + df_fc.Dose.astype(str)+ '_' + df_fc.Time + '_' + df_fc.Cluster
#     df_summary = df_fc.pivot(values='logFC',columns='condition').transpose()
#     offset = []
#     for i,tag in enumerate(['Drug','Dose','Time']) :
#         metadata = [x.split('_')[i] for x in df_summary.index]
#         df_summary.insert(0,tag,metadata)
#         offset+=['']
#     df_summary = df_summary.transpose()
#     for i, tag in enumerate(['Location','Marker']) :
#         metadata = [x.split('_')[i] for x in df_summary.index[3 :]]
#         df_summary.insert(0,tag,offset+metadata)
#     df_summary.to_csv('d :/data/MCF10A drug response heatmap.csv')

"""


def make_complex_heatmap(df_data, heatmap_cmap='coolwarm',
                         vmax=4,
                         vmin=-4,
                         figsize=(16, 9),
                         row_metadata=None,
                         col_metadata=None,
                         col_colorbar_anchor=[0.12, 0.1, 0.7, 0.05],
                         row_colorbar_anchor=[0.85, 0.15, 0.02, 0.7],
                         figname=None):
    """Script for making complex heatmaps with sidebars and legends of each colorbar.
        Row metadata are shown in additional columns of heatmap and col metadata are shown in additional rows.

    Parameters
    --------
    df_data : pd.DataFrame
        a table with data.
    row_metadata, col_metadata : pd.Series or pd.DataFrame
        metadata of rows and columns that needs to be represented as row and column sidebars.
    figsize : figure size as in matplotlib.
    vmax, vmin : : float
        max or min value in the heatmap function fron Seaborn
    col_colorbar_anchor, row_colorbar_anchor : list of length of 4
        coordinants and size of color bar, first two values are x and y co-ordinnants, third and fourth one are width and height.
    figname : str
        path of output figure, if None, print to console.
    """
    # Initialize subplots.
    row_metadata = pd.DataFrame(row_metadata)
    col_metadata = pd.DataFrame(col_metadata)
    n_row = row_metadata.shape[1] + 1
    n_col = col_metadata.shape[1] + 1
    height_ratios = [15] + [1] * (n_col - 1)
    width_ratios = [15] + [1] * (n_row - 1)
    fig, axes = plt.subplots(n_col, n_row, sharex=False, sharey=False, figsize=figsize, gridspec_kw={'height_ratios': height_ratios,
                                                                                                     'width_ratios': width_ratios,
                                                                                                     'wspace': 0.1,
                                                                                                     'hspace': 0})
    if n_row * n_col > 1:
        # Axes are flattened for easier indexing
        axes = axes.ravel()
        main_fig = sns.heatmap(df_data, vmax=vmax, vmin=vmin, ax=axes[
                               0], cbar=False, cmap=heatmap_cmap, robust=True)
    else:
        main_fig = sns.heatmap(
            df_data, vmax=vmax, vmin=vmin, cbar=False, cmap=heatmap_cmap, robust=True)
    # Make the main heatmap as the first subplot
    main_fig_axes = fig.add_axes([0.13, 0.95, 0.7, 0.05])
    main_fig_cb = plt.colorbar(main_fig.get_children()[
                               0], orientation='horizontal', cax=main_fig_axes)
    main_fig_cb.ax.set_title("Heatmap", position=(1.06, 0.1), fontsize=16)
    main_fig.set_xticks([])
    main_fig.set_yticks([])
    main_fig.set_ylabel(
        'logFC change compared with corresponding DMSO', fontsize=14)
    # Iterate through each metadata dataframe and start ploting the color bar
    # and heatmaps row-wise or column-wise
    for metadata, base_anchor, anchor_offset_location in zip([row_metadata, col_metadata], [row_colorbar_anchor, col_colorbar_anchor], [0, 1]):
        axes_offset = 1
        if metadata is None:
            continue
        # Iterate through each metadata colorbar
        for col in metadata.columns:
            metadata_vector = metadata[col]
            # Handling continuous heatmap sidebar values
            try:
                metadata_vector = metadata_vector.astype(float)
                metadata_vector = pd.DataFrame(metadata_vector, columns=[col])
                levels = metadata_vector[col].sort_values().unique()
                cmap = 'Blues'
                cb_type = 'continuous'
            # Handling descrete heatmap sidebar values, which are factorized.
            except ValueError:
                levels = metadata_vector.factorize()[1]
                metadata_vector = pd.DataFrame(
                    metadata_vector.factorize()[0], columns=[col])
                cmap = sns.color_palette("cubehelix_r", levels.shape[0])
                cb_type = 'discreet'

            # Calculate the axes index and location of the "legend" of the
            # sidebar, which are actually colorbar objects.
            if anchor_offset_location == 0:
                offset = 0.1
                # Column side bar offsets.
                ax = axes[axes_offset]
                cbar_label_orientation = 'vertical'
                cbar_title_location = (1.03, 1)
            else:
                offset = -0.1
                # Row side bar offsets.
                ax = axes[axes_offset * n_row]
                cbar_label_orientation = 'horizontal'
                cbar_title_location = (1.03, 0.1)
                metadata_vector = metadata_vector.transpose()

            # Plotting the sidebar and its colorbar
            anchor = base_anchor
            anchor[anchor_offset_location] = anchor[
                anchor_offset_location] + offset
            colorbar_ax = fig.add_axes(anchor)
            g = sns.heatmap(metadata_vector, ax=ax, cbar=False, xticklabels=False,
                            yticklabels=False, cmap=cmap, vmax=metadata_vector.values.max() + 1)
#             g.set_title(col)
            if cb_type != 'continuous':
                cb = plt.colorbar(
                    g.get_children()[0], orientation=cbar_label_orientation, cax=colorbar_ax)
                # Make correct ticks and tick labels, need to offset the lenth
                # to fix the miss-by-one problem.
                cb.set_ticks(np.arange(0.5, 0.5 + len(levels), 1))
                if anchor_offset_location == 0:
                    cb.ax.set_yticklabels(levels.values, fontsize=14)
                else:
                    cb.ax.set_xticklabels(levels.values, fontsize=14)
            else:
                cb = plt.colorbar(
                    g.get_children()[0], orientation=cbar_label_orientation, cax=colorbar_ax)
            cb.ax.set_title(col, position=cbar_title_location, fontsize=14)
            cb.ax.invert_yaxis()
            # To the next subplot axes
            axes_offset += 1
    # Get rid of empty subplots not used in the figure.
    valid_axes_id = [x for x in range(
        n_col)] + [x * n_row for x in range(n_col)]
    for axes_id in range(len(axes)):
        if axes_id not in valid_axes_id:
            fig.delaxes(axes[axes_id])

    # This is a hack in order to make the correct X axis label
    axes[n_row * (n_col - 1)].set_xlabel('Treatments', fontsize=14)
    if figname is not None:
        plt.savefig(figname, bbox_inches='tight')
        plt.close()


def check_numeric(str_value):
    if not isinstance(str_value, str):
        return str_value
    if str_value.isdigit():
        return int(str_value)
    elif str_value.replace('.', '').isdigit():
        return float(str_value)
    else:
        return str_value
