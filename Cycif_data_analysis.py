import pandas as pd
import numpy as np
import os
import random
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
from common_apis import *


class per_well_analysis():

    '''
    Analysis focusing on difference between target well against reference well (control).
    '''

    def __init__(self, expr_data, metadata,
                 output_path='results',
                 output_prefix='MCF10A Commons ',
                 time_col='time',
                 drug_col='ligand',
                 dose_col='dose',
                 batch_col=None,
                 control_name='PBS',
                 control_col=None):
        '''
        Differential analysis comparing each well to their corresponding controls. Need a unique control sample for each treatment condition. 
        For data with multiple batches/places, the analysis is done independantly within each batch.
        The way this is implemented is through defining a list of trt and control pairs over all conditions and run tests based on each individual contrast.     
        Parameters
        ========
        expr_data: pd.DataFrame, sample x features. dataframe of segmented single cell cycif data. 
        metadata: pd.DataFrame, sample x features, dataframe of metadata of segmented single cell data.
        output_path: str, as named.
        output_prefix: str, as named.
        time_col: str, column in metadata containing trt time duration.
        drug_col: str, column in metadata containing trt drug name.
        control_name: str, name of ligand used as control.
        control_col: str, column in metadata containing control sample. This is needed since it could interesting to use a time point as control rather than a ligand/drug.
        batch_col: str, column name in the metadata. This is usefult because sometimes we have pooled data from multiple plate/group, and each has their own control samples.
            separating proper control for proper test set is then important. By default their is no batches.
        '''

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
        '''
        Evaluate channel distributions and total variance of data
        '''
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
        '''
        Extract pairs of teatment sample set and control sample set. Assumes there is only one appropriate control for each treatment group.
        These sets were used in later analysis such as differential analysis and distance evaluation.
        This is done through grouping the cells based on the a collection of metadata such as drug, dose, and time. 
        Thus if a particular condition is of no interest to the question, it can be simply dropped.

        Parameters
        ========
        batch_wise: boolean, if True, make contrasts within each batch, otherwise assume there is only one control in the data.
        test_condition_groupby: list-like, collections of colnames that defines a treatment condition.  
        control_name: str, name of control sample name. 
        control_col_idx: str, name of column in metadata that contains the control name.

        Return
        ========
        list_groups: list, each element is a tuple of test_samples and control_samples.

        '''
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
                print('Control condition not found!')
                print('This is all conditions in batch {}:{}:'.format(
                    self.batch_col, str(batch)))
                print(current_batch_agg)
                raise ValueError()
            # Make actually contrasts
            for idx in current_batch_agg.index:
                if idx[control_col_idx] == control_name:
                    continue
                else:
                    batch_contrasts.append(('_'.join([str(x) for x in idx]), current_batch_agg.loc[
                                           idx], control_cells_metadata, control_cells, batch))
            self.list_all_contrasts += batch_contrasts

    def contrast_based_distance(self):
        '''
        Measure distances of treated group to control groups, and plot violinplots of distance.
        Distance to control is defined as euclidean distance of each treated sample to centroid of control.

        Return
        ========
        df_dist: pd.DataFrame, a dataframe of sample X features. trt to control distance is only one of the columns.
            The rest are sample metadata to allow flexibility in making the plots.
        '''
        time_col = self.time_col
        drug_col = self.drug_col
        dose_col = self.dose_col
        control_name = self.control_name
        df_dist = pd.DataFrame()

        for contrast in self.list_all_contrasts:
            test_samples = contrast[1].split(',')
            control_samples = contrast[3].split(',')
            test_metadata = contrast[0].split('_')
            control_metadata = contrast[2]
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
        '''
        Make violin plots based on different possible grouping methods

        Parameters
        ========
        facetgrid_col: str, column in df_dist, defines how the subplots are organized by column.   
        x_group: str, column in df_dist, defines how the groups are created in each subplot. 
        facetgrid_row: str, column in df_dist, similar to facetgrid_col, but for rows. 
        '''
        save_fig_filename = os.path.join(self.output_path, ' '.join(
            ['Drug_control distance over', x_group, 'per', facetgrid_col]) + '.png')
        sns.set(font_scale=1.5)
        order = sorted(self.metadata[x_group].unique())
        g = sns.FacetGrid(data=self.df_dist, col=facetgrid_col, row=facetgrid_row,
                          palette='Blues', sharey=True, sharex=True, height=6)
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
        '''
        Differential analysis comparing the first set of samples vs the second set of samples.
        '''
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
        '''
        Make data matrix of logFC values of each marker in each condition. 
        The actual matrix can be saved somewhere for future analysis in Morpheus. 

        Parameters
        ========
        condition_by: list-like or None, list of metadata that combined makes one-to-one well to condition mapping.
            If a condition describes multiple wells, the function will not run until provided with a valid condition_by value. 
        '''
        df_fc = self.report.copy()
        # Evaluate condition_by value.
        if condition_by is None:
            condition_by = [self.drug_col, self.dose_col, self.time_col]
        df_fc['condition'] = df_fc[condition_by].astype(
            str).apply(lambda x: '_'.join(x), axis=1)

        if df_fc.groupby('condition').logFC.count()[0] != df_fc.index.unique().shape[0]:
            no_replicates = (df_fc.groupby('condition').logFC.count()[
                             0]) / (df_fc.index.unique().shape[0])
            print('There seem to be multiple values ({}) for the same condition. Consider modify the "condition_by" paramerter!"'.format(
                str(no_replicates)))
            return
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
                2:].astype(int).apply(lambda x: "{0:0=2d}h".format(x))
        except:
            pass
        if output_fname is not None:
            output_name = os.path.join(
                self.output_path, self.output_prefix + 'logFC heatmap.csv')
            df_fc.to_csv(output_name)

        self.fc_heatmap = df_fc
        self.fc_heatmap_col_meta = condition_by

    def make_heatmaps(self, sorting='2D_HC'):
        # plot FC heatmap
        n_metacols = len(self.fc_heatmap_col_meta)
        heatmap_cmap = sns.palettes.diverging_palette(
            240, 0, s=99, as_cmap=True)
        df_data = self.fc_heatmap.iloc[n_metacols:, 2:].astype(float)
        output_fname = os.path.join(self.output_path, ' '.join(
            [self.output_prefix, sorting, 'logFC heatmap.png']))
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
        if sorting == '2D_HC':
            # 2D sorted heatmap
            fig = mch(df_data)
        else:
            # simple sorted heatmap
            cols_order = self.fc_heatmap.iloc[:, 2:].sort_values(
                self.fc_heatmap_col_meta, axis=1).columns
            row_order = self.fc_heatmap.iloc[
                n_metacols:, :].sort_values(['Location', 'Marker']).index
            df_data = df_data.loc[row_order, cols_order]
            row_meta = row_meta.loc[row_order]
            col_meta = col_meta.loc[cols_order]
            fig = mch(df_data)

    def make_swarm_plot(self, color_by=None, x_group=None):
        '''
        Plate-wise swarm plot of logFC per channel/feature, with flexibility in coloring and grouping along x-axis
        '''
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
        '''
        Make channel distribution plot for each channel/feature. X-axis separated by time and col separated by ligand
        Conditions where the channel was differentially expressed passing fc_threshold are represented with solid lines, otherwise dotted.
        '''
        if organize_by is None:
            organize_by = [self.time_col, self.drug_col]
        if not os.path.exists(self.output_path + '/misc'):
            os.mkdir(self.output_path + '/misc')
        # Use independant plotting functinos instead of those in the class.
        grouped_fc_table, reformed_expr_data, line_style = channel_dstrn_plot_data_prep(
            self.report, self.expr_data, self.metadata, fc_threshold=fc_threshold, organize_by=organize_by)
        _ = channel_dstrn_plot(grouped_fc_table, reformed_expr_data, organize_by=organize_by,
                               control_name=self.control_name, output_prefix=self.output_prefix, line_style=line_style)

        # # Plotting
        # sns.set(font_scale=1.5)
        # for channel in grouped_fc_table.index.unique():
        #     channel_logfc = grouped_fc_table.loc[channel]
        #     channel_data = reformed_expr_data.loc[channel]
        #     line_style = channel_logfc.groupby(organize_by).line_style.first()
        #     g = sns.FacetGrid(channel_data,col=self.time_col, hue=self.drug_col, palette="Set2",sharey = False ,sharex=True,height = 5)
        #     g = (g.map(sns.distplot, "Log2 Cycif intensity", hist=False, rug=False))
        #     g.add_legend(markerscale = 2,title=channel)
        #     lines = []
        #     for sub_plot in g.axes.flatten():
        #         time = check_numeric(sub_plot.get_title().split(' = ')[1])
        #         for line in sub_plot.lines:
        #             ligand = line.get_label()
        #             if ligand == self.control_name:
        #                 continue
        #             line.set_linestyle(line_style[(time,ligand)])

        #     output_fname = ' '.join([self.output_prefix, channel.replace('/', '+'), 'logFC distribution overtime.png']) # some channel names have '/' in it, sooo irritating!!!
        #     output_fname = os.path.join(self.output_path,'misc',output_fname)
        #     plt.savefig(output_fname)
        #     plt.close()
        # sns.set(font_scale=1)
