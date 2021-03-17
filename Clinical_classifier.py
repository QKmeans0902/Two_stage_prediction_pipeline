import os.path as osp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix


"""----------------------User Configuration----------------------"""
fpath = 'path/storing/your/features/files(.txt)'
lpath = 'path/storing/your/labels/files(.txt)'

x = np.loadtxt(fpath, delimiter='\t')
if x.ndim < 2:
    x = x.reshape(-1, 1)
y = np.squeeze(np.genfromtxt(lpath))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)
nested_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
eval_metrics = np.zeros((skf.n_splits, 3))

for n_fold, (train, test) in enumerate(skf.split(x, y)):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    log_clf_init = LogisticRegression()
    grid = GridSearchCV(log_clf_init, {'C': np.logspace(-3, 4, 8)}, cv=nested_skf, scoring='balanced_accuracy', n_jobs=5)
    grid.fit(x_train, y_train)
    log_clf = LogisticRegression()
    log_clf.fit(x_train, y_train)
    y_pred = log_clf.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fold_sensitivity = tp / (tp + fn)
    fold_specificity = tn / (tn + fp)
    fold_balanced_accuracy = (fold_sensitivity + fold_specificity) / 2
    eval_metrics[n_fold, 0] = fold_sensitivity
    eval_metrics[n_fold, 1] = fold_specificity
    eval_metrics[n_fold, 2] = fold_balanced_accuracy

df = pd.DataFrame(eval_metrics)
df.columns = ['SEN', 'SPE', 'BAC']
df.index = ['Fold_' + str(i + 1) for i in range(skf.n_splits)]
print(df)
print('\nAverage Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
