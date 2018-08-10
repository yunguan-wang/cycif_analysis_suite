import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import fdrcorrection as fdr
from matplotlib import pyplot as plt
from umap import UMAP
from sklearn.cluster import DBSCAN
import hdbscan

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
    report = pd.DataFrame(columns=['logFC', 'pValue', 'adj_pVal'])
    for feature in control_norm.columns:
        fc = test_norm[feature].mean() - control_norm[feature].mean()
        pval = ttest(control_norm[feature],test_norm[feature])
        report.loc[feature,'logFC'] = np.round(fc,2)
        report.loc[feature,'pValue'] = pval[1]
    report['adj_pVal'] = fdr(report.pValue)[1]
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