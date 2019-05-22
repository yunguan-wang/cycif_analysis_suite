import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from adjustText import adjust_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import robust_scale, quantile_transform
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.mixture import GaussianMixture


def dimension_reduction(roi_data, n_pc=10, umap_n_neighbor_modifier=500, cols=None, umap_nc=2):
    np.random.RandomState(seed=999)
    if cols is None:
        cols = roi_data.columns
    roi_data = roi_data[cols]
    pca = PCA(n_pc)
    if roi_data.shape[0] // umap_n_neighbor_modifier < 15:
        nn = 15
    else:
        nn = roi_data.shape[0] // umap_n_neighbor_modifier
    umap = UMAP(n_neighbors=nn, n_components=umap_nc)
    roi_pca = pca.fit_transform(scale(roi_data))
    roi_pca = pd.DataFrame(roi_pca, index=roi_data.index, columns=[
                           'pc' + str(x) for x in range(1, 1 + roi_pca.shape[1])])
    roi_umap = umap.fit_transform(roi_pca)
    # get rid of outliers
    roi_umap = pd.DataFrame(roi_umap, index=roi_data.index)
    lof = LocalOutlierFactor(n_neighbors=roi_umap.shape[
                             0] // 50, contamination='auto', metric='euclidean')
    labels = lof.fit_predict(roi_umap)
    valid_idx = [i for i, x in enumerate(labels) if x != -1]
    roi_umap = roi_umap.iloc[valid_idx]
    roi_pca = roi_pca.iloc[valid_idx]
    roi_data = roi_data.iloc[valid_idx]
    return roi_data, roi_pca, roi_umap


def clustering_wrapper(data, algorithm='hdbscan', dimention_reduction=False, n_clusters=2, mcs=None,
                       ms=None, umap_n_comp=2, umap_min_dist=0.1, umap_n_neighbors=15,
                       use_full_dim=True, **kwargs):
    '''
    wrapper function for GMM, KMEANS and HDBSCAN clustering.
    '''
    if algorithm == 'gmm':
        clustering = GaussianMixture(n_components=n_clusters)
        clustering.fit(data)
        labels = clustering.predict_proba(data)
        labels = np.argmax(labels, axis=1)
    else:
        if dimention_reduction == 'umap':
            umap = UMAP(n_components=umap_n_comp, min_dist=umap_min_dist,
                        n_neighbors=umap_n_neighbors, **kwargs)
            transformed_data = umap.fit_transform(data)
        elif dimention_reduction == 'pca':
            pca = PCA(0.8)
            transformed_data = pca.fit_transform(data)
        elif dimention_reduction == 'pca+umap':
            pca = PCA(0.95)
            umap = UMAP(n_components=umap_n_comp, min_dist=umap_min_dist,
                        n_neighbors=umap_n_neighbors, **kwargs)
            transformed_data = umap.fit_transform(pca.fit_transform(data))
        else:
            transformed_data = None
        if algorithm == 'hdbscan':
            if mcs == None:
                mcs = int(0.1 * data.shape[0])
            if ms == None:
                ms = int(0.1 * data.shape[0])
            clustering = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        elif algorithm == 'kmeans':
            clustering = KMeans(n_clusters)
        if use_full_dim:
            labels = clustering.fit_predict(data)
        else:
            labels = clustering.fit_predict(transformed_data)
    return labels, transformed_data


def cluster_all_wells(expr_data, metadata, clustering_func, group_id='sampleid', mcs=5,
                      verbose=True, output_dstn=None, **kwargs):
    '''
    Cluster cells in each cell.
    '''
    if expr_data.max().max() > 1000:
        expr_data = np.log2(expr_data)
    current_cwd = os.getcwd()
    if output_dstn is not None:
        os.chdir(output_dstn)
    metadata['cluster'] = 0
    for sample_group in metadata.groupby(group_id):
        group_name = sample_group[0]
        group_idx = sample_group[1].index
        group_df = expr_data.loc[group_idx]
        if group_df.shape[0] < 200:
            continue
        if verbose:
            print('Sample name: {}'.format(group_name))
        labels, umap_data = clustering_func(group_df)
        metadata.loc[group_idx, 'cluster'] = labels
        umap_data = pd.DataFrame(umap_data[:, :2], columns=[
                                 'UMAP_C1', 'UMAP_C2'])
        sns.scatterplot('UMAP_C1', 'UMAP_C2', palette='Set2', hue=labels,
                        data=umap_data, s=5, legend='full', edgecolor=None, alpha=0.8)
        plt.savefig(group_name + '.png')
        plt.xlim(umap_data.UMAP_C1.quantile(0.01),
                 umap_data.UMAP_C1.quantile(0.99))
        plt.ylim(umap_data.UMAP_C2.quantile(0.01),
                 umap_data.UMAP_C2.quantile(0.99))
        plt.close()
    os.chdir(current_cwd)
    return metadata


def cluster_DEG(df, labels, num_tops=3, threshold=0.2):
    avg_expr = df.groupby(labels).mean()
    fc_expr = pd.DataFrame(index=np.unique(labels), columns=df.columns)
    for label in np.unique(labels):
        if label == -1:
            continue
        ref_idx = [x != label or x != -1 for x in labels]
        ref_idx = df.index[ref_idx]
        fc_expr.loc[label] = avg_expr.loc[label] - df.loc[ref_idx].mean()
        tops = fc_expr.sort_values(label, axis=1, ascending=False).loc[label]
        tops = tops[tops >= threshold].index[:num_tops].values
        fc_expr.loc[label, 'tops'] = ', '.join(tops)
    return fc_expr.dropna()


def batch_normlization(data, metadata, batch_col='plate'):
    """log2 transformation, combat batch effect removal, and robust scaling for data. 
        grouping_crit: list, column names specifying grouping condition. last one must be the cluster column name.

    Parameters
    --------
    method: str
        data normalization methods, can be 'combat' or 'median'.

    Returns
    --------
    norm_data: pandas.DataFrame
        batch effect normalized expr data.
    norm_meta: pandas.DataFrame
        index metched metadata of norm_data.
    """
    if data.index.shape != metadata.index.shape:
        print('Index mismatch!Attempted to correct it.')
        cm_idx = [x for x in data.index if x in metadata.index]
        data = data.loc[cm_idx].copy()
        metadata = metadata.loc[cm_idx].copy()
    # log2 transformation and batch effect removal.
    norm_data = np.log2(data)
    norm_data = combat(norm_data.transpose(), batch=metadata[
                       batch_col]).transpose()
    norm_meta = metadata.loc[norm_data.index]
    return norm_data, norm_meta


def per_group_normalization(expr, metadata, grouping_crit=['plate', 'ROI', 'cluster'], norm_fn='robust_scale'):
    if isinstance(grouping_crit, list):
        grouping_crit = metadata[grouping_crit].apply(
            lambda x: '_'.join(x.astype(str)), axis=1)
    else:
        grouping_crit = metadata[grouping_crit]
    if expr.max().max() > 1000:
        expr = np.log2(expr.copy())
    grouped_expr = expr.groupby(grouping_crit)

    def _norm_func(group_series, norm_fn):
        if norm_fn == 'robust_scale':
            norm_func = robust_scale(
                group_series.values.reshape(-1, 1), quantile_range=(1, 99))[:, 0]
        elif norm_fn == 'median':
            norm_func = group_series - group_series.median()
        elif norm_fn == 'quantile':
            norm_func = quantile_transform(
                group_series.values.reshape(-1, 1))[:, 0]
        else:
            raise ValueError(
                'norm_fn must be one of the following: robust_scale, median, power or quantile')
        return norm_func

    norm_expr = grouped_expr.transform(_norm_func, norm_fn)
    return norm_expr


def bin_expr_data(expr_data, metadata, bin_on_col='grouping_crit', batch=None):
    '''
    bin expr data based on grouping keys provided in the metadata. 
    Parameters
    ========
    batch: dict, dictionary of batch column and batch value, to allow binning of only a subset of data.
    '''
    if sorted(metadata.index) != sorted(expr_data.index):
        print("Input expr data and metadata do not have matching indices!")
    if batch is not None:
        batch_col = list(batch.keys())[0]
        batch_val = batch[batch_col]
        metadata_idx = metadata[metadata[batch_col] == batch_val].index
    else:
        metadata_idx = metadata.index
    cm_idx = [x for x in expr_data.index if x in metadata_idx]
    grouping_keys = metadata.loc[cm_idx, bin_on_col].values
    expr_data = expr_data.loc[cm_idx]
    metadata = metadata.loc[cm_idx]
    binned_expr_data = expr_data.groupby(grouping_keys).mean()
    return binned_expr_data


def group_wise_channel_dist_plot(expr, metadata, channel=['ECad', 'SMA', 'Vimentin', 'CK8'], grouping_id='group_id', fig_name='well-wise channel dist.pdf'):
    plot_data = expr[channel].reset_index().melt(
        id_vars='index', var_name='channel').set_index('index')
    plot_data.loc[:, grouping_id] = metadata.loc[plot_data.index, grouping_id]
    plot_data.sort_values(grouping_id, inplace=True)
    g = sns.FacetGrid(plot_data, hue='channel', col=grouping_id,
                      col_wrap=4, height=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value', hist=False)
    for subplot in g.axes:
        subplot.legend()
    g.savefig(fig_name)
    plt.close()


def PCA_transform_data(binned_expr_data, metadata,
                       num_top_pca_genes=4,
                       binned_on_col='sampleid',
                       metadata_cols=['ligand', 'time']
                       ):
    # PCA transformation of binned tata
    pca = PCA(2)
    pca.fit(binned_expr_data)
    transformed_data = pca.transform(binned_expr_data)
    transformed_data = pd.DataFrame(transformed_data, columns=[
                                    'X1', 'X2'], index=binned_expr_data.index)
    binned_metadata = metadata.drop_duplicates(binned_on_col)
    binned_metadata = binned_metadata.set_index(binned_on_col)[metadata_cols]
    transformed_data = transformed_data.merge(
        binned_metadata, how='left', left_index=True, right_index=True)
    # Get top genes based on PCA loadings.
    pcc = pca.components_
    pcc = pd.DataFrame(pcc, columns=binned_expr_data.columns,
                       index=['pc1', 'pc2']).transpose()
    top_genes = []
    for pc in pcc.columns:
        top_genes += abs(pcc).sort_values(pc,
                                          ascending=False).index.tolist()[:num_top_pca_genes]
    top_genes = list(set(top_genes))
    top_genes = pcc.loc[top_genes]
    top_genes.loc['evr', 'pc1'] = pca.explained_variance_ratio_[0]
    top_genes.loc['evr', 'pc2'] = pca.explained_variance_ratio_[1]
    return transformed_data, top_genes


def plot_PCA_2D(transformed_data, top_pca_genes,
                metadata_cols=['ligand', 'time'], arrow_anchor=(12, 6),
                arrow_length_fold=3, figname=None, add_loading_arrows=True,
                figsize=(16, 8),
                **kwargs):
    plt.figure(figsize=figsize)
    for i, col in enumerate(metadata_cols):
        if col not in transformed_data.columns:
            metadata_cols[i] = None
    color_col = metadata_cols[0]
    if len(metadata_cols) > 1:
        size_col = metadata_cols[1]
    else:
        size_col = None
    g = sns.scatterplot('X1', 'X2', data=transformed_data, hue=color_col,
                        size=size_col, sizes=(15, 450), **kwargs)
    # adding PC arrows for top genes
    if add_loading_arrows:
        for gene in top_pca_genes.index:
            loading1 = top_pca_genes.loc[gene, 'pc1']
            loading2 = top_pca_genes.loc[gene, 'pc2']
            if gene == 'evr':
                continue  # skip explained variance ratio row
            arrow_x_length = arrow_length_fold * \
                top_pca_genes.loc[gene, 'pc1']
            arrow_y_length = arrow_length_fold * \
                top_pca_genes.loc[gene, 'pc2']
            g.arrow(arrow_anchor[0], arrow_anchor[1],
                    arrow_x_length, arrow_y_length,
                    color='r', alpha=0.5, width=0.02)
            g.text(arrow_anchor[0] + arrow_x_length,
                   arrow_anchor[1] + arrow_y_length,
                   '{}: {:.2f}, {:.2f}'.format(gene, loading1, loading2),
                   ha='center', va='center', fontsize=12)
        annotations = [x for x in g.get_children() if type(x) ==
                       matplotlib.text.Text]
        adjust_text(annotations)
    # labeling x and y axies
    evr_pc1 = top_pca_genes.loc['evr', 'pc1']
    evr_pc2 = top_pca_genes.loc['evr', 'pc2']
    plt.xlabel('PC1 : {:.2}% variance explained'.format(
        str(100 * evr_pc1)), fontsize=14)
    plt.ylabel('PC2 : {:.2}% variance explained'.format(
        str(100 * evr_pc2)), fontsize=14)
    _ = plt.legend(bbox_to_anchor=(1., 0.5), loc='center left', fontsize=16)
    # for handle in lgnd.legendHandles:
    #     handle._sizes = [150]
    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname, dpi=600)
        plt.close()
    else:
        return g


def expr_pca_plot(transformed_data, binned_expr_data, ncols=4, size_col='time', figname=None):
    nrows = int(np.ceil(binned_expr_data.shape[1] / ncols))
    figsize = (6 * ncols, 3 * nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()
    i = 0
    size_time = 2 * np.log2(transformed_data[size_col] + 1)
    # size_time = 75*transformed_data.cluster_size.values
    for col in binned_expr_data.columns:
        g = axes[i].scatter(transformed_data.X1, transformed_data.X2, cmap='coolwarm',
                            c=binned_expr_data[col], s=size_time, edgecolor='face')
        axes[i].set_title(col, fontsize=20)
        axes[i].set_facecolor('gray')
        cb = fig.colorbar(g, ax=axes[i])
        cb.ax.tick_params(labelsize=20)
        i += 1
    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.close()
