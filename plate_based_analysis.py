import pandas as pd
import numpy as np
import os
import random
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
from common_apis import *


def dot_heatmap(plotdata, x=1, y=0, figsize=(32, 12), size_scale=30, fontsize=18, half_ticks=False, figname=None):
    """Make dotted heatmap based on given long-format data.

    Parameters:
    --------
    plotdata: pd.DataFrame
        a table of plot data. First two columns corresponds to the y and x axies, and the third is fold change.
    x: int
        column index of x axis of the plot.
    y: int
        column index of y axis of the plot.
    figsize: tuple
        width and height of the figure.
    size_scale: int
        adjusts the sizes of dots in the heatmap.
    fontsize: int
        x and y label font size.
    half_ticks: bool
        option to show only half of the x axis labels.
    figname: None or str
        output figure name, if None, figure will be printed to console.

    """
    fig, ax = plt.subplots(figsize=figsize)
    y_labels = plotdata.iloc[:, y].unique()
    plotdata['sign'] = plotdata.iloc[:, 2] > 0
    colorbar_base_anchor = [0.88, 0.15, 0.02, 0.7]
    plotdata['x'] = plotdata.iloc[:, x].factorize()[0]
    plotdata['y'] = plotdata.iloc[:, y].factorize()[0]
    for if_positive, cmap in zip([True, False], ['Reds', 'Greens']):
        _plotdata = plotdata[plotdata.sign == if_positive]
        size_vector = size_scale * abs(_plotdata.iloc[:, 2])
        color_vector = abs(_plotdata.iloc[:, 2])
        x_vector = _plotdata.x
        y_vector = _plotdata.y
        ax.scatter(x_vector, y_vector, s=size_vector,
                   cmap=cmap, c=color_vector)
        if cmap == 'Reds':
            cbar_ax = fig.add_axes(colorbar_base_anchor)
            cb = plt.colorbar(ax.get_children()[0], ax=ax, cax=cbar_ax)
        else:
            colorbar_base_anchor[0] = colorbar_base_anchor[0] + 0.04
            cbar_ax = fig.add_axes(colorbar_base_anchor)
            cb = plt.colorbar(ax.get_children()[1], ax=ax, cax=cbar_ax)
        # adjusting color bar labels.
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(16)
    # adjust x and y ticks label fonts and rotations.
    ax.tick_params(axis='x', labelrotation=75, width=5, length=15)
    if half_ticks:
        channels = plotdata.iloc[:, x].apply(
            lambda x: x.split('_')[1]).unique()
        ax.set_xticks(np.arange(1, 2 * len(channels), 2))
    else:
        channels = plotdata.iloc[:, x].unique()
        ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, fontsize=fontsize, fontdict={
                       'horizontalalignment': 'right'})
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=fontsize)
    ax.tick_params('y', labelsize=16)
    ax.set_facecolor('white')
    if figname is not None:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()


def channel_dstrn_plot_data_prep(df_fc, expr_data, metadata, fc_threshold=0.6, organize_by=['ligand', 'dose', 'time']):
    """Prep the data for 'channel_dstrn_plot' function. 
        Need the fold change table created by the per_well_analysis module.
    """
    # Determines line styles of each condition
    df_fc['ch'] = df_fc.index
    # Since there are replicates, the mean logFC for each channel is used
    grouped_fc_table = df_fc.groupby(
        organize_by + ['ch']).logFC.mean().reset_index()
    # TODO: implement col sep by dose option
    grouped_fc_table.sort_values(organize_by, inplace=True)
    grouped_fc_table['line_style'] = '-'
    grouped_fc_table.loc[abs(grouped_fc_table.logFC)
                         < fc_threshold, 'line_style'] = ':'
    line_style = grouped_fc_table.groupby(
        organize_by + ['ch']).line_style.first()
    grouped_fc_table = grouped_fc_table.set_index('ch')
    channels = grouped_fc_table.index.unique()
    # Reformate expr data into long format
    reformed_expr_data = expr_data[channels].reset_index()
    reformed_expr_data = reformed_expr_data.melt(
        'index', var_name='Channel',  value_name='Log2 Cycif intensity')
    reformed_expr_data.set_index('index', inplace=True)
    reformed_expr_data = reformed_expr_data.merge(
        metadata[organize_by], left_index=True, right_index=True, sort=False, how='left')
    reformed_expr_data.sort_values(organize_by, inplace=True)
    reformed_expr_data.set_index('Channel', inplace=True)
    return grouped_fc_table, reformed_expr_data, line_style


def channel_dstrn_plot(reformed_expr_data, line_style, hue_sep='time', col_sep='ligand', control_name='PBS', output_prefix='', output_folder='Results/misc', colwrap=None, savefig=True):
    """Standalone function for making channel distribution plot for each channel/feature. 
        Conditions where the channel was differentially expressed passing fc_threshold are represented with solid lines, otherwise dotted.

    Parameters
    --------
    df_fc: pd.DataFrame
        a dataframe object returned by the 'channel_dstrn_plot_data_prep' function. 
    reformed_expr_data: pd.DataFrame
        a dataframe object of the input data, same as the one used in per_well_analysis.
    line_style: pd.DataFrame
        a look up table of line style for each feature + experimental condition combination. 
        This is one of the outputs from the 'channel_dstrn_plot_data_prep' function.
    hue_sep: str
        column name in df_fc dataframe that defines each line in the distribution plot.
    col_sep: str
        column name in df_fc dataframe that defines each subplots in the plot.
    control_name: str
        value for the control sample name.
    """
    facetgrid_col = col_sep
    # channels = df_fc.index.unique()
    channels = reformed_expr_data.index.unique()
    list_channel_plots = []
    sns.set(font_scale=1.5)
    line_style = line_style.reset_index().set_index('ch')
    for channel in channels:
        channel_line_style = line_style.loc[channel]
        channel_line_style = channel_line_style.groupby(
            [col_sep, hue_sep]).line_style.first()
        # channel_logfc = df_fc.loc[channel]
        channel_data = reformed_expr_data.loc[channel]
        # set line style
        g = sns.FacetGrid(channel_data, col=facetgrid_col, col_wrap=colwrap,
                          hue=hue_sep, palette="Set2", sharey=False, sharex=True, height=5)
        g = (g.map(sns.distplot, "Log2 Cycif intensity", hist=False, rug=False))
        g.add_legend(markerscale=2, title=channel)
        lines = []
        for sub_plot in g.axes.flatten():
            key1 = check_numeric(sub_plot.get_title().split(' = ')[1])
            if key1 == control_name:
                continue
            for line in sub_plot.lines:
                key2 = line.get_label()
                if key2 == control_name:
                    continue
                line.set_linestyle(channel_line_style[
                                   (check_numeric(key1), check_numeric(key2))])
        if savefig:
            # some channel names have '/' in it, sooo irritating!!!
            output_fname = ' '.join([output_prefix, channel.replace(
                '/', '+'), 'logFC distribution overtime.png'])
            output_fname = os.path.join(output_folder, output_fname)
            plt.savefig(output_fname)
            plt.close()
        list_channel_plots.append(g)
    sns.set(font_scale=1)
    return list_channel_plots


class per_well_analysis():

    """Analysis focusing on difference between target well against reference well (control).
    """

    def __init__(self, expr_data, metadata,
                 output_path='results',
                 output_prefix='MCF10A Commons ',
                 time_col='time',
                 drug_col='ligand',
                 dose_col='dose',
                 batch_col=None,
                 control_name='PBS',
                 control_col=None):
        """Differential analysis comparing each well to their corresponding controls. 
            Need a unique control sample for each treatment condition. 
            For data with multiple batches/places, the analysis is done independantly within each batch. 
            The way this is implemented is through defining a list of trt and control pairs over all conditions and run tests based on each individual contrast.     

        Parameters
        --------
        expr_data: pd.DataFrame
            sample x features. dataframe of segmented single cell cycif data. 
        metadata: pd.DataFrame
            sample x features, dataframe of metadata of segmented single cell data.
        output_path: str
            as named.
        output_prefix: str
            as named.
        time_col: str
            column in metadata containing trt time duration.
        drug_col: str
            column in metadata containing trt drug name.
        control_name: str
            name of ligand used as control.
        control_col: str
            column in metadata containing control sample. This is needed since it could interesting to use a time point as control rather than a ligand/drug.
        batch_col: str
            column name in the metadata. This is usefult because sometimes we have pooled data from multiple plate/group, and each has their own control samples.
            separating proper control for proper test set is then important. By default their is no batches.
        """

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if control_col is None:
            control_col = drug_col
        expr_data = expr_data.loc[metadata.index]
        self.expr_data = expr_data
        self.metadata = metadata
        self.output_path = output_path
        self.output_prefix = output_prefix
        self.drug_col = drug_col
        self.dose_col = dose_col
        self.time_col = time_col
        self.control_name = control_name
        self.control_col = control_col
        self.batch_col = batch_col

    def descriptive_analysis(self,
                             return_variance=False,
                             channel_dist_plot=True,
                             fig_name='Channel distribution plot.png'):
        """Evaluate channel distributions and total variance of data. 
            Includes histograms for each channel and well-wise variance table.

        Parameters
        --------
        return_variance: bool
            if True, variance table are returned.
        channel_dist_plot: bool
            if True, make channel histograms.

        Returns
        total_variance: well-wise variance table.

        """
        # Channel distribution plot.
        drug_group = self.metadata[self.drug_col]
        dose_group = self.metadata[self.dose_col]
        time_group = self.metadata[self.time_col]
        if channel_dist_plot:
            fig_name = os.path.join(self.output_path, ' '.join(
                [self.output_prefix, fig_name]))
            channel_histograms(self.expr_data, save_fig_filename=fig_name)
        # Evaluate total variance.
        total_variance = self.expr_data.groupby(
            [drug_group, dose_group, time_group]).std().sum(axis=1)
        if return_variance:
            return total_variance
        else:
            total_variance_fn = os.path.join(self.output_path, ' '.join(
                [self.output_prefix, 'log2 transformed data total variance by condition.csv']))
            total_variance.to_csv(total_variance_fn)

    def make_contrast(self, batch_wise=False, test_condition_groupby=None, control_name=None, control_col_idx=0):
        """Extract pairs of teatment sample set and control sample set. 
            Assumes there is only one appropriate control for each treatment group.
            These sets were used in later analysis such as differential analysis and distance evaluation.
            This is done through grouping the cells based on the a collection of metadata such as drug, dose, and time. 
            Thus if a particular condition is of no interest to the question, it can be simply dropped.

        Parameters
        --------
        batch_wise: boolean
            if True, make contrasts within each batch, otherwise assume there is only one control condition in the data.
        test_condition_groupby: list-like
            collections of colnames that defines a treatment condition.  
        control_name: str
            name of control sample name. 
        control_col_idx: str
            index in 'test_condition_groupby' speciying the name of column in metadata that contains the control name.

        Return
        --------
        list_groups: list
            a tuple of test condition, test_samples, control condition and control_samples.

        """
        # setting default as drug + dose + time
        if test_condition_groupby is None:
            test_condition_groupby = [
                self.drug_col, self.dose_col, self.time_col]
        if control_name is None:
            control_name = self.control_name
        self.list_all_contrasts = []
        self.contrast_by = test_condition_groupby
        # define all batches
        if batch_wise:
            list_batches = sorted(self.metadata[self.batch_col].unique())
        else:
            list_batches = ['single_batch']
        # define controls
        control_col = test_condition_groupby[control_col_idx]
        if control_name != self.control_name:
            print('Control sample changed to {}:{} from {}:{}!'.format(
                control_col, control_name, self.control_col, self.control_name))

        # batch-wise operation, each batch works with only samples of current
        # batch.
        for batch in list_batches:
            batch_contrasts = []
            if batch == 'single_batch':
                current_batch_samples = self.metadata.index
            else:
                current_batch_samples = self.metadata[
                    self.metadata[self.batch_col] == batch].index
            current_batch = self.metadata.loc[current_batch_samples]
            current_batch_agg = current_batch.groupby(test_condition_groupby).agg(
                lambda x: ','.join(x.index)).iloc[:, 0]

            # This loop is to identify control and pad control cells and control metadata to each test condition in current batch.
            # TODO: It could be improved.
            _status = 0
            for idx in current_batch_agg.index:
                if idx[control_col_idx] == control_name:
                    control_cells = current_batch_agg.loc[idx]
                    control_cells_metadata = '_'.join([str(x) for x in idx])
                    _status = 1

            # if no controls were found, throw an exception and return some
            # information on why it is so.
            if _status == 0:
                print('This is all conditions in batch {}:{}:'.format(
                    self.batch_col, str(batch)))
                print(current_batch_agg)
                raise ValueError('Control condition not found!')
            # Make actually contrasts
            for idx in current_batch_agg.index:
                if idx[control_col_idx] == control_name:
                    continue
                else:
                    batch_contrasts.append(('_'.join([str(x) for x in idx]), current_batch_agg.loc[
                                           idx], control_cells_metadata, control_cells, batch))
            self.list_all_contrasts += batch_contrasts

    def contrast_based_distance(self):
        """Measure distances of treated group to control groups, and plot violinplots of distance.
            Distance to control is defined as euclidean distance of each treated sample to centroid of control.

        Return
        --------
        df_dist: pd.DataFrame
            a dataframe of sample by features. treatment to control distance is only one of the columns.
            The rest are sample metadata to allow flexibility in making the plots.
        """
        df_dist = pd.DataFrame()

        for contrast in self.list_all_contrasts:
            test_samples = contrast[1].split(',')
            control_samples = contrast[3].split(',')
            test_metadata = contrast[0].split('_')
            test_expr = self.expr_data.loc[test_samples]
            control_expr = self.expr_data.loc[control_samples]
            # get "centroid" by find the median control sample.
            control_vector = control_expr.median().values.reshape(1, -1)
            dist_vector = euclidean_distances(control_vector, test_expr)
            # clipping outliers out.
            low = np.percentile(dist_vector, 1)
            high = np.percentile(dist_vector, 99)
            dist_vector = dist_vector[
                (dist_vector > low) & (dist_vector < high)]
            dist_vector = pd.DataFrame(dist_vector, columns=['distance'])
            # Fillin metadata
            for idx, metadata_col in enumerate(self.contrast_by):
                metadata_value = check_numeric(test_metadata[idx])
                dist_vector[metadata_col] = metadata_value
            df_dist = df_dist.append(dist_vector)
        self.df_dist = df_dist

    def plot_dist_violinplot(self, facetgrid_col, x_group, facetgrid_row=None):
        """Make violin plots based on different possible grouping methods.

        Parameters
        --------
        facetgrid_col: str
            column in df_dist, defines how the subplots are organized by column.   
        x_group: str
            column in df_dist, defines how the groups are created in each subplot. 
        facetgrid_row: str
            column in df_dist, similar to facetgrid_col, but for rows. 

        """
        save_fig_filename = os.path.join(self.output_path, ' '.join(
            ['Drug_control distance over', x_group, 'per', facetgrid_col]) + '.png')
        sns.set(font_scale=1.5)
        order = sorted(self.metadata[x_group].unique())
        g = sns.FacetGrid(data=self.df_dist, hue='ligand', col=facetgrid_col,
                          row=facetgrid_row, sharey=True, sharex=True, height=6)
        g = g.map(sns.violinplot, x_group, 'distance', order=order, bw=.2)
        # rotate x-axis labels
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(60)
        plt.tight_layout()
        plt.savefig(save_fig_filename)
        plt.close()
        sns.set(font_scale=1)

# plot_dist_violinplot(df_dist,x_axis_col='Time',save_fig_filename=output_prefix+' Treatment to control distance.png')

    def contrast_based_differential_analysis(self, save_report=True):
        """Differential analysis comparing the first set of samples vs the second set of samples.
        """
        Report = pd.DataFrame()
        for batch in self.list_all_contrasts:
            test_samples = batch[1].split(',')
            control_samples = batch[3].split(',')
            test_metadata = batch[0].split('_')
            control_metadata = batch[2]
            batch_id = batch[4]
            batch_de_report = differential_analysis(
                self.expr_data.loc[test_samples], self.expr_data.loc[control_samples])

            # add metadata
            for idx, metadata_col in enumerate(self.contrast_by):
                metadata_value = check_numeric(test_metadata[idx])
                batch_de_report[metadata_col] = metadata_value

            batch_de_report['Abs_logFC'] = abs(batch_de_report.logFC)
            batch_de_report['Batch'] = batch_id
            batch_de_report['Control_metadata'] = control_metadata
            Report = Report.append(batch_de_report)
            Report.logFC = Report.logFC.astype(float)
        self.report = Report
        if save_report is True:
            output_fname = os.path.join(self.output_path, ' '.join(
                [self.output_prefix, 'logFC report.xlsx']))
            Report.to_excel(output_fname)

    def make_fc_heamtap_matrix(self, condition_by=None, output_fname=None):
        """Make data matrix of logFC values of each marker in each condition. 
            The actual matrix can be saved somewhere for future analysis in Morpheus. 

        Parameters
        --------
        condition_by: None or list-like
            list of metadata that combined makes one-to-one well to condition mapping.
            If None, the conditions are defined by drug, dose and time.
            If a condition describes multiple wells, the function will not run until provided with a valid condition_by value. 
        """
        df_fc = self.report.copy()
        # Evaluate condition_by value.
        if condition_by is None:
            condition_by = [self.drug_col, self.dose_col, self.time_col]
        df_fc['condition'] = df_fc[condition_by].astype(
            str).apply(lambda x: '_'.join(x), axis=1)

        if df_fc.groupby('condition').logFC.count()[0] != df_fc.index.unique().shape[0]:
            num_reps = (df_fc.groupby('condition').logFC.count()[
                0]) / (df_fc.index.unique().shape[0])
            raise ValueError('There seem to be multiple values ({}) for the same condition. Consider modify the "condition_by" paramerter!"'.format(
                str(num_reps)))
        # Make a matrix of channel/feature by conditions.
        df_fc = df_fc.pivot(values='logFC', columns='condition').transpose()
        # Order samples and features using 2d hierarchical clustering.
        df_fc = df_fc.iloc[two_way_hc_ordering(df_fc)].astype(float)
        # Adds metadata.
        offset = []
        for i, tag in enumerate(condition_by):
            metadata = [x.split('_')[i] for x in df_fc.index]
            df_fc.insert(0, tag, metadata)
            offset += ['']
        df_fc = df_fc.transpose()
        for i, tag in enumerate(['Location', 'Marker']):
            metadata = [x.split('_')[i] for x in df_fc.index[3:]]
            df_fc.insert(0, tag, offset + metadata)

        # Reformat time information into two digit strings
        try:
            df_fc.loc[self.time_col][2:] = df_fc.loc[self.time_col][
                2:].astype(int).apply(lambda x: "{0:0=2d}".format(x))
        except:
            pass
        if output_fname is not None:
            output_name = os.path.join(
                self.output_path, self.output_prefix + 'logFC heatmap.csv')
            df_fc.to_csv(output_name)

        self.fc_heatmap = df_fc
        self.fc_heatmap_col_meta = condition_by

    def make_dotted_heatmap(self, **kwargs):
        """Wrapper for makiing dotted heatmap.
        """
        plotdata = (self.fc_heatmap).copy()
        batch = plotdata.loc['ligand'] + '_' + plotdata.loc['time']
        plotdata.sort_values(['Marker', 'Location'], inplace=True)
        plotdata = plotdata.sort_values(['ligand', 'time'], axis=1)
        plotdata = plotdata.iloc[3:, 2:]
        plotdata = plotdata.astype(float)
        plotdata = plotdata.transpose().groupby(batch, sort=False).median()
        plotdata = plotdata.stack().reset_index()
        figname = os.path.join(
            self.output_path, self.output_prefix + 'logFC dotted heatmap.png')
        dot_heatmap(plotdata, figname=figname, **kwargs)

    def make_heatmaps(self, style='2D_HC'):
        """Make logFC heatmaps out of folc change table. 
            If style is '2D_HC', the heatmap will be 2D hierarchical ordered.
            Otherwise it will be sorted simply based on drug/dose/time. 
        """
        # plot FC heatmap
        n_metacols = len(self.fc_heatmap_col_meta)
        heatmap_cmap = sns.palettes.diverging_palette(
            240, 0, s=99, as_cmap=True)
        df_data = self.fc_heatmap.iloc[n_metacols:, 2:].astype(float)
        output_fname = os.path.join(self.output_path, ' '.join(
            [self.output_prefix, style, 'logFC heatmap.png']))
        row_meta = self.fc_heatmap.iloc[n_metacols:, :2]
        col_meta = self.fc_heatmap.iloc[:n_metacols, 2:].transpose()

        # wrapper for passing parameters.
        def mch(data):
            make_complex_heatmap(data,
                                 heatmap_cmap=heatmap_cmap,
                                 row_metadata=row_meta,
                                 col_metadata=col_meta,
                                 col_colorbar_anchor=[0.12, 0.1, 0.7, 0.05],
                                 row_colorbar_anchor=[0.81, 0.16, 0.02, 0.7],
                                 figname=output_fname)
            return

        # Write eigher 2d HC sorted heatmap or simple sorted heatmap.
        if style == '2D_HC':
            # 2D sorted heatmap
            mch(df_data)
        else:
            # simple sorted heatmap
            cols_order = self.fc_heatmap.iloc[:, 2:].sort_values(
                self.fc_heatmap_col_meta, axis=1).columns
            row_order = self.fc_heatmap.iloc[
                n_metacols:, :].sort_values(['Location', 'Marker']).index
            df_data = df_data.loc[row_order, cols_order]
            row_meta = row_meta.loc[row_order]
            col_meta = col_meta.loc[cols_order]
            mch(df_data)

    def make_swarm_plot(self, color_by=None, x_group=None):
        """Plate-wise swarm plot of logFC per channel/feature, with flexibility in coloring and grouping along x-axis.
        """
        sns.set(font_scale=2)
        if color_by is None:
            color_by = self.drug_col
        if x_group is None:
            x_group = self.time_col
        output_fname = os.path.join(self.output_path, ' '.join(
            [self.output_prefix, 'swarm logFC plot by', x_group, '.png']))
        df_fc = self.report.copy()
        # Assumes the first two '_' separated elements of index are location
        # and channel name
        df_fc['Ch'] = ['_'.join(x.split('_')[:2]) for x in df_fc.index]
        df_fc.sort_values(x_group, inplace=True)
        g = sns.FacetGrid(data=df_fc, col='Ch', hue=color_by,
                          col_wrap=5, height=5, aspect=2)
        g = g.map(sns.swarmplot, x_group, 'logFC', s=10)
        g.add_legend(markerscale=2)
        plt.savefig(output_fname)
        plt.close()
        sns.set(font_scale=1)

    def channel_dist_plot(self, fc_threshold=0.6, organize_by=None, savefig=True):
        """Make channel distribution plot for each channel/feature. 
            Generate a figure for each channel. Each figure has multiple subplots in columns.
            Columns are separated by time and colors are assigned according to ligand, by default.
            Conditions where the channel was differentially expressed passing fc_threshold are represented with solid lines, otherwise dotted.

        Parameters
        --------
        fc_threshold: float
            Fold change threshold for considering if the channel's distribution will be plotted as a solid line.
        organize_by: None or list-like
            Determines the layout of the figure. The first value for columns separator, and the second value for color.
        savefig: bool
            If true, figures will be saved as ./misc/[prefix+channel name]. 

        """
        if organize_by is None:
            organize_by = [self.time_col, self.drug_col]
        if not os.path.exists(self.output_path + '/misc'):
            os.mkdir(self.output_path + '/misc')
        # Use independant plotting functinos instead of those in the class.
        grouped_fc_table, reformed_expr_data, line_style = channel_dstrn_plot_data_prep(
            self.report, self.expr_data, self.metadata, fc_threshold=fc_threshold, organize_by=organize_by)
        _ = channel_dstrn_plot(reformed_expr_data, hue_sep=organize_by[1], col_sep=organize_by[0],
                               control_name=self.control_name, output_prefix=self.output_prefix, line_style=line_style)
