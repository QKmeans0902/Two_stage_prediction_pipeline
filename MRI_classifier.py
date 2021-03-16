import xlrd
import pandas as pd
import os.path as osp
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix

from Utils import *
from Model import FeatureReduction


n_train = 10000
h_units = [128, 32, 4, 1]
dpath = 'path/storing/your/features/and/labels/files'
feature_npath = 'path/of/feature/name/files'

x = np.loadtxt(osp.join(dpath, 'features.txt'), delimiter=',')
y = np.squeeze(np.genfromtxt(osp.join(dpath, 'labels.txt')))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)
nested_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
fold_contribution = np.zeros((x.shape[1], skf.n_splits))
eval_metrics = np.zeros((skf.n_splits, 3))

for n_fold, (train, test) in enumerate(skf.split(x, y)):

    """Binary processing and pre-training SAE"""
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    x_train_bin, x_test_bin = binarization(x_train, x_test, y_train)
    total_AE_weight, total_AE_bias = sae_pretraining(x_train_bin, h_units)
    median_weight, median_bias = median_init(total_AE_weight), median_init(total_AE_bias)

    """Training Feed Forward Network"""
    fr_nn = FeatureReduction(x_train_bin.shape[0], h_units, median_weight, median_bias)
    optimizer = optim.Adam(fr_nn.parameters(), lr=0.01)
    nn_weights = train(fr_nn, optimizer, x_train_bin, y_train)
    ldc_train, ldc_test = nn_ldc(fr_nn, x_train_bin), nn_ldc(fr_nn, x_test_bin)

    """SVM classifier"""
    svm_init = SVC(kernel='linear')
    grid = GridSearchCV(svm_init, {'C': np.logspace(-3, 4, 8)}, cv=nested_skf, scoring='balanced_accuracy', n_jobs=5)
    grid.fit(ldc_train, ldc_train)
    svm = SVC(C=grid.best_params_['C'], kernel='linear')
    svm.fit(ldc_train, ldc_train)
    y_pred = svm.predict(ldc_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fold_sensitivity = tp / (tp + fn)
    fold_specificity = tn / (tn + fp)
    fold_balanced_accuracy = (fold_sensitivity + fold_specificity) / 2
    eval_metrics[n_fold, 0] = fold_sensitivity
    eval_metrics[n_fold, 1] = fold_specificity
    eval_metrics[n_fold, 2] = fold_balanced_accuracy

    """Calculating feature contributions to the dimensionality reduction"""
    selected_node = np.array(range(h_units[-1]))
    nn_weights.reverse()
    for weight in nn_weights:
        selected_weight = weight[selected_node]
        contribution = np.sum(np.abs(selected_weight), axis=0) / np.sum(np.abs(selected_weight))
        descend_idx = np.argsort(-contribution)
        num_selected_node = np.sum(np.cumsum(-np.sort(-contribution)) < .5) + 1
        selected_node = descend_idx[range(num_selected_node)]
    fold_contribution[:, n_fold] = contribution

"""Print classification results"""
df = pd.DataFrame(eval_metrics)
df.columns = ['BAC', 'SEN', 'SPE']
df.index = ['Fold_' + str(i + 1) for i in range(skf.n_splits)]
print(df)
print('\nAverage Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))

"""Save & print top 10 brain regions"""
region_list = xlrd.open_workbook(feature_npath).sheet_by_index(0)
average_contribution = np.mean(fold_contribution, axis=1)
top_order = np.argsort(-average_contribution)
top10_ind = top_order[:10]
with open(dpath + '/top10.txt', 'w+') as top10:
    for index in top10_ind:
        top10.write(region_list.cell(index, 0).value + '\n')

