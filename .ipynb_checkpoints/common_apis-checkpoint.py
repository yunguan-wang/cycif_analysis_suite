import pandas as pd
import numpy as np
import os
import seaborn as sns
from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import fdrcorrection as fdr
from matplotlib import pyplot as plt
from umap import UMAP
from sklearn.cluster import DBSCAN
import hdbscan

def channel_histograms(df, xlim = None,save_fig_filename = None):
    sns.set(font_scale=3)
    plt.rcParams['patch.linewidth'] = 0
    plt.rcParams['patch.edgecolor'] = 'none'
    n_rows = int(np.ceil(df.columns.shape[0]/4))
    _, axs = plt.subplots(n_rows,4, sharex=True)
    if xlim is None:
        pass
    else:
        plt.xlim(xlim)
    axs = axs.ravel()
    for i,col in enumerate(df.columns):
        df[col].hist(bins = 100,ax = axs[i],figsize = (64,27), label = col,grid = False)
        axs[i].set_title(col, x=0.2, y=0.6)
    if save_fig_filename is None:
        plt.show()
    else:
        plt.savefig(save_fig_filename)
    plt.close()
    
def preprocessing_log_transform(df_data):
    # setting a size threshold for cell size to get rid of doublets
    area_cols = [x for x in df_data.columns if 'Area' in x]
    invalid_cells = []
    for area_col in area_cols:
        cell_size_threshold_upper = df_data[area_col].median() + 4*df_data[area_col].std()
        invalid_cells += df_data[df_data[area_col]>cell_size_threshold_upper].index.tolist()
    invalid_cells = list(set(invalid_cells))
    df_data = df_data.drop(invalid_cells)
    df_data = df_data.iloc[:,4:]
    df_data = np.log2(df_data+1)
    return df_data, invalid_cells

def plot_clustering(df_low_dim, df_labels, figname = None):
    """
    Make scatter plots of dimention reduced data with cluster designation
    """
    plt.ioff()
    df_labels = df_labels.reset_index()
    for cluster in sorted(df_labels.iloc[:,1].unique()):
        cells_idx = df_labels[df_labels.iloc[:,1]==cluster].index.values
        plt.scatter(df_low_dim[cells_idx,0],df_low_dim[cells_idx,1],label=cluster, s = 1)
    
    plt.legend(markerscale=5,bbox_to_anchor=(1, 0.9))
    if figname is None:
        plt.savefig('Clustering_on_2D.png',bbox_inches='tight')
    else:
        plt.savefig(figname,bbox_inches='tight')
    plt.close()

def differential_analysis(test_norm, control_norm):
    '''
    calculated fold change on log transformed data.
    ========
    Paremeters:
    test_norm: pandas.dataframe, a dataframe of normalized test data, rows as cells and columns as features.
    control_norm: pandas.dataframe, a dataframe of normalized control data, rows as cells and columns as features.
    '''
    report = pd.DataFrame(columns=['logFC', 'T_pValue','KS_pValue', 'adj_T_pVal', 'adj_KS_pVal'])
    for feature in control_norm.columns:
        fc = test_norm[feature].mean() - control_norm[feature].mean()
        pval = ttest(control_norm[feature],test_norm[feature])
        ks_pval = ks_2samp(control_norm[feature],test_norm[feature])[1]
        report.loc[feature,'logFC'] = np.round(fc,2)
        report.loc[feature,'T_pValue'] = pval[1]
        report.loc[feature,'KS_pValue'] = ks_pval
    report['adj_T_pVal'] = fdr(report.T_pValue)[1]
    report['adj_KS_pVal'] = fdr(report.KS_pValue)[1]
    return report

def cluster_based_DE(test_norm, control_norm, figname):
    data_umap = umap.fit_transform(test_norm)
    labels = clustering_function.fit_predict(data_umap)
    labels = pd.Series(['Cluster ' + str(x) for x in labels],index=test_norm.index)
    overall_report = pd.DataFrame()
    for cluster in labels.unique():
        if cluster == 'Cluster -1':
            continue
        current_cluster_cells = labels[labels==cluster].index
        if len(current_cluster_cells) <= 0.05*len(test_norm):
            continue
        print('\tPerforming differential analsis on {} which has {} cells'.format(cluster, str(len(current_cluster_cells))))
        test_current_cluster = test_norm.loc[current_cluster_cells]
        report_current_cluster = differential_analysis(test_current_cluster,control_norm)
        report_current_cluster['Cluster'] = cluster
        overall_report = overall_report.append(report_current_cluster)
    plot_clustering(data_umap,labels,figname)
    return overall_report, labels

def plot_expr_on_2D(df_2d,df_raw_expr,figname,labels):
    """
    Make a grid of scatter plots of original data projected to a 2D space determined by UMAP.
    Each marker is visualized in a subplot and each point in the subplot is colored based on its original expression. 
    """
    plt.ioff()
    nrows = int(np.ceil(1+len(df_raw_expr.columns)/5))
    fig, ax = plt.subplots(nrows, 5, sharex='col', sharey='row',squeeze=False,figsize=(25,5*nrows))
    ax = ax.ravel()
    
    # plot control vs test
    labels.index = list(range(len(labels)))
    for label in labels.unique():
        idx = labels[labels==label].index
        ax[0].scatter(df_2d[idx,0],df_2d[idx,1], cmap='bwr',s = 1,label = label)
    ax[0].legend(markerscale=5, frameon=False)
    
    # plot markers
    idx = 1
    for colname in df_raw_expr.columns:
        subplot = ax[idx].scatter(df_2d[:,0],df_2d[:,1],c = df_raw_expr[colname].values, s = 1,cmap='bwr',label = colname)
        ax[idx].legend(markerscale=5, frameon=False)
        fig.colorbar(subplot,ax=[ax[idx],ax[idx]],pad=-0.05,extend='both',format='%.1f')
        idx+=1
    
    plt.savefig(figname,bbox_inches='tight')
    plt.close()
    
def umap_parameter_opt():
    random_idx = np.random.choice(df_pooled.index,10000, False)
    sub_df = df_pooled.loc[random_idx].iloc[:,:2]
    list_nn = [5,15,25,35,45,55,65,75,85]
    lsit_md = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    _,axes = plt.subplots(ncols=3, nrows=3,figsize = (32,18))
    axes = axes.ravel()
    idx = 0
    for n_neighbors in list_nn:
        umap = UMAP(n_neighbors = n_neighbors,n_components=2,min_dist=0.1, metric='correlation', random_state=50)
        df_pooled_time_umap = umap.fit_transform(sub_df)
        metadata = pd.read_csv('MCF10A commons metadata.csv',index_col=0)
        df_pooled_time_umap = pd.DataFrame(df_pooled_time_umap,columns = ['X' + str(i) for i in range(df_pooled_time_umap.shape[1])])
        axes[idx].scatter(df_pooled_time_umap.X0, df_pooled_time_umap.X1, s=0.01)
        axes[idx].set_title('n_neighbors: {}'.format(str(n_neighbors)))
        idx+=1

    _,axes = plt.subplots(ncols=3, nrows=3,figsize = (32,18))
    axes = axes.ravel()
    idx = 0    
    for md in lsit_md:
        umap = UMAP(n_neighbors = 5,n_components=2,min_dist=md, metric='correlation',random_state=50)
        df_pooled_time_umap = umap.fit_transform(sub_df)
        metadata = pd.read_csv('MCF10A commons metadata.csv',index_col=0)
        df_pooled_time_umap = pd.DataFrame(df_pooled_time_umap,columns = ['X' + str(i) for i in range(df_pooled_time_umap.shape[1])])
        axes[idx].scatter(df_pooled_time_umap.X0, df_pooled_time_umap.X1, s=0.01)
        axes[idx].set_title('min_dist: {}'.format(str(md)))
        idx+=1