import xlrd
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix

from Utils import *
from Model import FeatureReduction


"""----------------------User Configuration----------------------"""
n_train = 10000 # time of stacked autoecoders trainging for meidan weight and bias
h_units = [128, 32, 4, 1] # number of units in each hidden layer
fpath = 'path/storing/your/features/files(.txt)'
lpath = 'path/storing/your/labels/files(.txt)'
gpath = 'path/storing/files/indicating/treatment/group(.txt)'

x = np.loadtxt(fpath, delimiter='\t')
y = np.squeeze(np.genfromtxt(lpath))
nested_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
group = np.genfromtxt(gpath)
index, reverse_index = np.unique(group, return_inverse=True)
fold_contribution = np.zeros((x.shape[1], index.size))
eval_metrics = np.zeros((index.size, 3))

for n_fold in range(index.size):

    """Binary processing and pre-training SAE"""
    x_test, y_test = x[reverse_index == n_fold], y[reverse_index == n_fold]
    x_train, y_train = x[reverse_index != n_fold], y[reverse_index != n_fold]
    x_train_bin, x_test_bin = binarization(x_train, x_test, y_train)
    total_AE_weight, total_AE_bias = sae_pretraining(x_train_bin, h_units, n_train)
    median_weight, median_bias = median_init(total_AE_weight), median_init(total_AE_bias)

    """Training Feed Forward Network"""
    fr_nn = FeatureReduction(x_train_bin.shape[0], h_units, median_weight, median_bias)
    optimizer = optim.Adam(fr_nn.parameters(), lr=0.01)
    nn_weights = train_nn(fr_nn, optimizer, x_train_bin, y_train)
    ldc_train, ldc_test = nn_ldc(fr_nn, x_train_bin), nn_ldc(fr_nn, x_test_bin)

    """SVM classifier"""
    svm_init = SVC(kernel='linear')
    grid = GridSearchCV(svm_init, {'C': np.logspace(-3, 4, 8)}, cv=nested_skf, scoring='balanced_accuracy', n_jobs=5)
    grid.fit(ldc_train, y_train)
    svm = SVC(C=grid.best_params_['C'], kernel='linear')
    svm.fit(ldc_train, y_train)
    y_pred = svm.predict(ldc_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fold_sensitivity = tp / (tp + fn)
    fold_specificity = tn / (tn + fp)
    fold_balanced_accuracy = (fold_sensitivity + fold_specificity) / 2
    eval_metrics[n_fold, 0] = fold_sensitivity
    eval_metrics[n_fold, 1] = fold_specificity
    eval_metrics[n_fold, 2] = fold_balanced_accuracy

"""Print classification results"""
df = pd.DataFrame(eval_metrics)
df.columns = ['SEN', 'SPE', 'BAC']
df.index = ['Fold_' + str(i + 1) for i in range(index.size)]
print(df)
print('\nAverage Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))


