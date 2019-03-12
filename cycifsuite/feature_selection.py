import pandas as pd
import os
import numpy as np
import seaborn as sns
from cycifsuite.get_data import read_synapse_file
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, RFECV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from rfpimp import *


def data_preprocessing(path, fname, ligand_name=None, sample_size=None, excluding_cells=[]):
    '''Read data, drop 0 variance columns and impute missing values.
    '''
    imputer = SimpleImputer(strategy='median')
    x = pd.read_hdf(path + fname)
    zero_std_cols = x.columns[x.std() == 0]
    x = x.drop(zero_std_cols, axis=1)
    x.loc[:, :] = imputer.fit_transform(x)

    # Sampling easier feature selection
    if sample_size is not None:
        if isinstance(sample_size, float):
            sample_size = int(sample_size * x.shape[0])
        elif isinstance(sample_size, int):
            pass
        # get rid of cells not wanted
        sample_pool = [k for k in x.index if k not in excluding_cells]
        sample_idx = np.random.choice(
            x.index, size=sample_size)
        x = x.loc[sample_idx]
    # make y_vector
    y_vector = None
    if ligand_name is not None:
        y_vector = [ligand_name] * x.shape[0]
    return x, y_vector


def make_x_and_y(path, feature_files, sample_size=None, norm_func=minmax_scale,
                 binning=False, metadata=None, test_size=None):
    """Read data with the ability to sample it. data from each file from feature_files
    will be combined into a sample by feature table. Then the data will be normalized 
    with desired normalize function.

    Paramerters:
    --------
    sample_size : None or int
        if none, all data will be usedm otherwise a number of data will be randomly selected.
    norm_func : function
        a function object to be passed as the feature wise normlizing function applied to 
        appended data.
    binning : bool
        option to bin the samples based on wells.
    metadata : none or pd.DataFrame
        a table of metadata for the expr data, must have 'Well' column.
    test_size : none or float
        whether split processed data into training and test sets. split ratio is defined by
        this parameter. if none, no splitting.

    Returns:
    --------
    x_train, x_test, y_train, y_test : 
        training and test data with labels.
    x,y : 
        unsplited data and labels.

    """
    x = pd.DataFrame()
    y = []
    for fn in feature_files:
        print('processing {}'.format(fn))
        ligand = fn.split('_')[-1][:-4]
        _x, _y = data_preprocessing(
            path, fn, ligand_name=ligand, sample_size=sample_size)
        if binning:
            cm_cells = [k for k in _x.index if k in metadata.index]
            _x = _x.loc[cm_cells].groupby(
                metadata.loc[cm_cells, 'Well']).median()
            _y = [ligand] * _x.shape[0]
        if x.shape[0] == 0:
            x = _x
        else:
            cols = [k for k in x.columns if k in _x.columns]
            x = x[cols].append(_x[cols])
        y += _y
    x.loc[:, :] = norm_func(x)
    if test_size is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test
    else:
        return x, y


def selct_k_anova(x, y, num_features):
    """Anova based on sklearn selectKbest.
    """
    sk = SelectKBest(k=num_features)
    sk.fit(x, y)
    sk_fs = sk.get_support()
    sk_fs = x.columns[sk_fs]
    return sk_fs, sk


def plot_feature_var(x, y, sk, f_meta, f_meta_col='a_feature_type'):
    """Plot feature variance and anova scores. Need the fitted model from 
    select_k_anova.
    """
    sns.set(font_scale=2, style='white')
    all_features = x.columns
    cm_features = [x for x in all_features if x in f_meta.index]
    f_meta = f_meta.loc[cm_features].copy()
    _, axes = plt.subplots(ncols=2, figsize=(16, 9))
    axes = axes.ravel()
    # axes[0].hist(x.std(),bins=50, label='All')
    for feature_type in f_meta[f_meta_col].unique():
        feature_list = f_meta[f_meta[f_meta_col] == feature_type].index
        sns.kdeplot(x[feature_list].std(), ax=axes[0], shade=True,
                    vertical=False, label=feature_type)
    axes[0].set_title('Distribution of std across all samples')
    axes[0].set_xlabel('Standard deviation of scaled data', fontsize=18)

    anova = pd.Series(sk.scores_, index=x.columns)
    anova[anova > 10000] = 10000
    for feature_type in f_meta[f_meta_col].unique():
        feature_list = f_meta[f_meta[f_meta_col] == feature_type].index
        sns.kdeplot(anova[feature_list], ax=axes[1],
                    shade=True, label=feature_type)
    # axes[1].set_xscale('log')
    axes[1].set_title('ANOVA F scores across all samples')
    axes[1].set_xlabel('ANOVA F score', fontsize=18)
    plt.tight_layout()
    plt.savefig('Feature variance and ANOVA.png')
    plt.close()
    sns.set(font_scale=1, style='white')


def plot_confusion_matrix(y_ture, y_pred, save_fig_prefix=None):
    cm = confusion_matrix(y_ture, y_pred)
    cm = pd.DataFrame(cm, index=np.unique(y_pred), columns=np.unique(y_pred))
    cm = cm / cm.sum(axis=1)
    sns.heatmap(cm, cmap='coolwarm')
    _ = plt.yticks(rotation=0)
    if save_fig_prefix is not None:
        plt.tight_layout()
        plt.savefig(save_fig_prefix + ' confusion_matrix.png')
        plt.close()


def PCA_transform_data(binned_expr_data, metadata,
                       num_top_pca_genes=4,
                       binned_on_col='sampleid',
                       metadata_cols=['ligand']
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


def plot_binned_pca_plot(x_binned, y_binned, arrows=False, **kwarg):
    """PCA transformation and plotting, colors are assigned by y_binned.
    """
    x_binned_meta = pd.DataFrame(
        y_binned, index=x_binned.index, columns=['ligand'])
    x_binned_meta['sampleid'] = x_binned_meta.index
    transformed_data, tg = PCA_transform_data(
        x_binned, x_binned_meta, 2, metadata_cols=['ligand'])
    plot_PCA_2D(transformed_data, tg, [
        'ligand'], add_loading_arrows=arrows, **kwarg)


def plot_feature_wise_pca(x_binned, y_binned, f_meta,
                          col='a_feature_type', save_fig_prefix=False,
                          **kwarg):
    """Wrapper function for plotting PCA plots across multiple feature 
    categories.
    """
    # Get feature stats
    cm_cols = [x for x in x_binned.columns if x in f_meta.index]
    f_meta = f_meta.reindex(cm_cols)
    all_features = x_binned.columns
    f_categories = f_meta[col].unique()
    # PCA plots of each feature category
    for f_type in f_categories:
        feature_list = f_meta[f_meta[col] == f_type].index.tolist()
        num_features = len(feature_list)
        plot_binned_pca_plot(x_binned[feature_list], y_binned, **kwarg)
        plt.title('{} : {}'.format(f_type, num_features), fontsize=18)
        if save_fig_prefix:
            plt.tight_layout()
            plt.savefig(save_fig_prefix + ' PCA 2D plot ' +
                        f_type + ' features.png')
            plt.close()


def rf_feature_imp_plot(rf_model, feature_meta, rf_model_features,
                        save_fig_prefix=False, ylim=(0, 0.015),
                        feature_meta_cols=[
                            'a_feature_type', 'feature_chr', 'sublocation', 'marker'],
                        cumulative=False,
                        title=''):
    feature_meta = feature_meta.copy()
    rf_fi = pd.Series(rf_model.feature_importances_, index=rf_model_features)
    cm_features = [x for x in feature_meta.index if x in rf_model_features]
    feature_meta.loc[cm_features, 'imp'] = rf_fi[cm_features].values
    feature_meta.imp.fillna(0, inplace=True)
    for fi_meta_type in feature_meta_cols:
        if cumulative:
            plot_data = feature_meta.groupby(fi_meta_type).sum()
            plot_data = plot_data.sort_values('imp', ascending=False)
            x_axis_len = plot_data.shape[0]
            plt.figure(figsize=(x_axis_len * 2, 8))
            sns.barplot(x=plot_data.index, y='imp', data=plot_data)
            plt.ylabel('Random forest cumulative feature imp', fontsize=24)
        else:
            plot_data = feature_meta[fi_meta_type].dropna()
            x_order = feature_meta.groupby(fi_meta_type).quantile(
                0.75).sort_values('imp', ascending=False).index
            if fi_meta_type == 'marker':
                plot_data[:] = [
                    'dna' if 'dna' in x else x for x in plot_data.values]
            x_axis_len = plot_data.unique().shape[0]
            plt.figure(figsize=(x_axis_len * 2, 8))
            sns.boxplot(x=fi_meta_type, y='imp', data=feature_meta.loc[
                        plot_data.index], order=x_order)
            plt.ylabel('Random forest feature imp', fontsize=24)

        plt.xticks(fontsize=24, rotation=45, ha='right')
        plt.yticks(fontsize=20)
        plt.xlabel(fi_meta_type, fontsize=24)
        plt.title(title, fontsize=18)
        plt.ylim(ylim)
        if save_fig_prefix:
            plt.tight_layout()
            plt.savefig(save_fig_prefix +
                        ' Random forest feature importance by {}.png'.format(fi_meta_type))
            plt.close()


def ensembled_feature_selection(x_train, y_train, x_test, y_test, num_features=50):
    # Select K best
    print('SelectKbest...')
    sk_fs, _ = selct_k_anova(x_train, y_train, num_features)

    # Random forest
    print('Random forest...')
    full_model = RandomForestClassifier(100)
    full_model.fit(x_train, y_train)
    rf_fi = pd.Series(full_model.feature_importances_, index=x_train.columns)
    rf_fs = rf_fi.sort_values(ascending=False).index[:num_features].tolist()

    # Permutation, out-out-bag based feature importance
    # https://github.com/parrt/random-forest-importances
    rfcv_x_train, rfcv_x_test, rfcv_y_train, rfcv_y_test = train_test_split(
        x_train, y_train, test_size=0.20)
    print('Random forest cv...')
    rfcv = RandomForestClassifier(100, n_jobs=4)
    rfcv.fit(rfcv_x_train, rfcv_y_train)
    rfimp_fi = importances(
        rfcv, rfcv_x_test, pd.Series(rfcv_y_test))  # permutation
    rfpimp_fs = rfimp_fi.index[:num_features].tolist()

    # RFE feature importance with logistic regression and random forest
    print('RFE LR...')
    rfe_input_features = rf_fi.sort_values(ascending=False).index[:500]
    sampled_x = x_train[rfe_input_features]
    sampled_y = y_train
    estimator = estimator = LogisticRegression(
        solver='newton-cg', multi_class='auto')
    selector = RFE(estimator, n_features_to_select=num_features, step=25)
    selector = selector.fit(sampled_x, sampled_y)
    rfe_fs_lr = sampled_x.columns[selector.support_]

    print('RFE RF...')
    estimator = RandomForestClassifier(100)
    selector = RFE(estimator, n_features_to_select=num_features,
                   step=25)
    selector = selector.fit(sampled_x, sampled_y)
    rfe_fs_rf = sampled_x.columns[selector.support_]

    # summerize all features
    feature_ranks = pd.DataFrame(index=x_train.columns)
    for col, fs_list in zip(['Anova', 'RF', 'RF_cv', 'RFE_rf', 'RFE_lr'],
                            [sk_fs, rf_fs, rfpimp_fs, rfe_fs_rf, rfe_fs_lr]):
        feature_ranks.loc[fs_list, col] = 1
    feature_ranks.fillna(0, inplace=True)
    feature_ranks = feature_ranks.sum(axis=1).sort_values(ascending=False)
    feature_ranks = pd.DataFrame(feature_ranks, columns=['feature_rank'])
    feature_ranks['RF_cv_fi'] = rfimp_fi.loc[feature_ranks.index].values
    feature_ranks['RF_fi'] = rf_fi.loc[feature_ranks.index].values
    feature_ranks = feature_ranks.sort_values(
        ['feature_rank', 'RF_cv_fi', 'RF_fi'], ascending=False)
    best_features = feature_ranks.index.tolist()[:num_features]

    # Evaluate best featuers
    three_abv = feature_ranks[feature_ranks.feature_rank >= 3].index.tolist()
    four_abv = feature_ranks[feature_ranks.feature_rank >= 4].index.tolist()
    for col, fs_list in zip(['Anova', 'RF', 'RF_cv', 'RFE_rf', 'RFE_lr', 'Best', 'Shared >=3', 'Shared >=4'],
                            [sk_fs, rf_fs, rfpimp_fs, rfe_fs_rf, rfe_fs_lr, best_features, three_abv, four_abv]):
        sub_features = fs_list
        x_train_sub = x_train[sub_features]
        x_test_sub = x_test[sub_features]
        fs_model = RandomForestClassifier(100)
        fs_model.fit(x_train_sub, y_train)
        y_pred = fs_model.predict(x_test_sub)
        print('{}: Selected {} feature accuracy {:.2f}'.format(
            col, len(sub_features), accuracy_score(y_test, y_pred)))
    y_pred = full_model.predict(x_test)
    print('Full feature model accuracy {:.2f}'.format(
        accuracy_score(y_test, y_pred)))
    return best_features, rf_fi, rfimp_fi


def main(sample_size=1000):
    path = 'd:/data/MCF10A 090718 data/'
    os.chdir(path)
    feature_files = [x for x in os.listdir() if '_txt' in x]
    pooled_metadata = pd.read_csv(
        read_synapse_file('syn17902177'), index_col=0)
    x, y = make_x_and_y(feature_files, sample_size)
    x.loc[:, :] = robust_scale(x)
    # get beste features based on anova.
    sk_features = fs_selectKbest(x, y, 20)


def evaluate_features(x, y, features, estimater):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    x_train_sub = x_train[features]
    x_test_sub = x_test[features]
    print('Training data size: {}'.format(x_train.shape))
    print('Test data size : {}'.format(x_test_sub.shape))
    full_model = RandomForestClassifier(100)
    fs_model = RandomForestClassifier(100)
    full_model.fit(x_train, y_train)
    y_pred = full_model.predict(x_test)
    print('Full feature accuracy {}'.format(accuracy_score(y_test, y_pred)))
    fs_model.fit(x_train_sub, y_train)
    y_pred = fs_model.predict(x_test_sub)
    print('Selected feature accuracy {}'.format(
        accuracy_score(y_test, y_pred)))

if __name__ == '__main__':
    main()
