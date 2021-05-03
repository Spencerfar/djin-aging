import numpy as np
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser('Predict change')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()

def classifier_test(X, y, weights):

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats = 10, random_state = 0)
    
    acc = []
    auc = []
    parameters = []
    for train_index, test_index in rskf.split(X[0], y[0]):
        
        X_train, X_test, y_train, y_test = X[:,train_index], X[:,test_index], y[:,train_index], y[:,test_index]
        
        X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
        y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1])
        y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1])
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        classifier = LogisticRegression(max_iter = 500, penalty='none')
        classifier.fit(X_train, y_train)
        acc.append(accuracy_score(y_test, classifier.predict(X_test)))
    
    parameters = np.array(parameters)
        
    mean = np.mean(acc)
    lower = 2*mean - np.percentile(acc, q=97.5)
    upper = 2*mean - np.percentile(acc, q=2.5)
    
    return acc, (mean, lower, upper)

X = np.load('../Analysis_Data/Synthetic_classifier_data%d_epoch%d.npy'%(args.job_id, args.epoch))
y = np.load('../Analysis_Data/Synthetic_classifier_labels%d_epoch%d.npy'%(args.job_id, args.epoch)).astype(int)
mask = np.load('../Analysis_Data/Synthetic_classifier_mask%d_epoch%d.npy'%(args.job_id, args.epoch))
weights = np.load('../Analysis_Data/Synthetic_classifier_weights%d_epoch%d.npy'%(args.job_id, args.epoch))

for t in range(20):
    
    m = mask[:,t].sum(-1)
    
    acc, acc_stats = classifier_test(X[:,m>0,t], y[:,m>0], weights)
    print(t, acc_stats)
    np.save('../Analysis_Data/Classifier_accuracy_time%d_job_id%d_epoch%d_acc.npy'%(t, args.job_id, args.epoch), acc)
