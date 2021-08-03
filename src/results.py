import os
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
import itertools
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_curve, precision_score, confusion_matrix, auc
import matplotlib
import platform
if 'Linux' in platform.platform():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import h5py
import pickle

from src.order_matrix import order_matrix
from src.BKSFusion import BKSFuser

sns.set_style("whitegrid", {'axes.grid' : False})
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}
font2 = {'size'   : 13}
plt.rc('font', **font2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)

    where_are_NaNs = np.isnan(cm)
    cm[where_are_NaNs] = 0.0
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', **font)
    plt.xlabel('Predicted label', **font)   

def get_labels(db, db_config):
    """
    Given the dataset and the database configuration
    return the possible classes
    """
    if db in ['controlled', 'uncontrolled']:
        if db_config == 1:
            classes = ['FB', 'FL', 'TW']
        elif db_config == 2:
            platforms = ['FB', 'FL', 'TW']
            classes = platforms + [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
        elif db_config == 3:
            platforms = ['FB', 'FL', 'TW']
            classes = [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
            classes =  platforms + classes + [p1+'-'+ p2 for p1 in classes for p2 in platforms]
    return classes

def get_classes_lbl(classes, subset, conf=None):
    """
    Given the input classes, the subset in which we are interested, and
    (when needed) the dataset configuration, return the new lables as
    a dictionary of strings that maps to an integer and the names of the
    new classes
    """
    lbls = {}

    if subset is None:
        for i, cl in enumerate(classes):
            lbls[cl] = i
        n_lbls = len(classes)
        new_cl = classes[:]

    elif subset == '1vsN':
        for cl in classes:
            if '-' in cl:
                lbls[cl] = 1
            else:
                lbls[cl] = 0

        new_cl=['NS', 'R1orR2']


    elif subset == 'last':
        tmp = [cl for cl in classes if '-' not in cl]
        new_cl = ['L-'+cl for cl in tmp]

        for cl in classes:
            if cl in tmp:
                lbls[cl] = tmp.index(cl)
            else:
                x = cl.split('-')[-1]
                lbls[cl] = tmp.index(x)

    elif 'last+' in subset:
        sh = subset.split('+')[1]

        for cl in classes:
            if cl.endswith(sh):
                lbls[cl] = cl.count('-')
            else:
                lbls[cl] = -1

        if conf == 2:
            new_cl=['NS-'+sh, 'R1-'+sh]
        elif conf == 3:
            new_cl=['NS-'+sh, 'R1-'+sh, 'R2-'+sh]


    elif (subset == 'nsr1') or (subset == 'nsr1r2') or ('nsr1r2+' in subset):
        for cl in classes:
            lbls[cl] = cl.count('-')
        if conf == 2:
            new_cl=['NS', 'R1']
        elif conf == 3:
            new_cl=['NS', 'R1', 'R2']

    elif 'nsr1' in subset and '+' in subset :
        new_cl = []
        sh = subset.split('+')[1]
        new_cl = [cl for cl in classes if cl.endswith(sh)]
        for cl in classes:
            if cl.endswith(sh):
                lbls[cl] = new_cl.index(cl)
            else:
                lbls[cl] = -1

    elif 'NS':
        new_cl = [cl for cl in classes if '-' not in cl]
        for cl in classes:
            if '-' in cl:
                lbls[cl] = -1
            else:
                lbls[cl] = new_cl.index(cl)


    return lbls, new_cl    

def discard(features, labels):
    """
    Discard features and labels when the latter is set to -1
    """
    features_new = []
    labels_new = []

    for f, l in zip(features, labels):
        if l != -1:
             features_new.append(f)
             labels_new.append(l)

    return features_new, labels_new

def balance(features, labels):
    """
    Balance features and labels lists to have the same
    number of elements for each possible class
    """
    features_new = []
    labels_new = []

    C = Counter(labels)
    min_c = C[min(C)]

    for f, l in zip(features, labels):
        if labels_new.count(l) < min_c:
            features_new.append(f)
            labels_new.append(l)

    return features_new, labels_new


def compute_svm(features_train, labels_train, features_test):
    """
    Given a list of train features, the train labels and the test features
    train an SVM and return the predictions for the feature vector
    """
    clf_svm = svm.LinearSVC(C=1)
    ovr_clf_svm = OneVsRestClassifier(clf_svm, n_jobs=-1)
    ovr_clf_svm.fit(features_train, labels_train)
    y_predict = ovr_clf_svm.predict(features_test)
    return y_predict

def compute_lr(features_train, labels_train, features_test):
    """
    Given a list of train features, the train labels and the test features
    train a LR model and return the predictions for the feature vector
    """
    clf_lr = OneVsRestClassifier(linear_model.LogisticRegression(C=1e5),n_jobs=-1)
    clf_lr.fit(features_train, labels_train)
    y_predict = clf_lr.predict(features_test)
    return y_predict


def compute_rf(features_train, labels_train, features_test, est):
    """
    Given a list of train features, the train labels and the test features
    train an Random Forest classifier and return the predictions for the feature vector
    """
    clf_rf = ensemble.RandomForestClassifier(n_estimators=est, random_state=0)
    ovr_clf_rf = OneVsRestClassifier(clf_rf, n_jobs=-1)
    ovr_clf_rf.fit(features_train, labels_train)
    y_predict = ovr_clf_rf.predict(features_test)
    return y_predict

def compute_rfb(features_train, labels_train, features_test, est, class_weight='balanced'):
    """
    Given a list of train features, the train labels and the test features
    train an Random Forest classifier and return the predictions for the feature vector
    """
    clf_rf = ensemble.RandomForestClassifier(n_estimators=est, random_state=1, class_weight=class_weight)
    ovr_clf_rf = OneVsRestClassifier(clf_rf, n_jobs=-1)
    ovr_clf_rf.fit(features_train, labels_train)
    y_predict = ovr_clf_rf.predict(features_test)
    return y_predict, ovr_clf_rf  # EDITED



def zero_div(a, b):
    """
    Return the floating point division of a and b.
    When b is zero return zero
    """
    try:
        c = float(a)/float(b)
        return c
    except:
        return 0

def save_res(y_pred, y_true, N, categories, fname, version, rej = None):
    """
    Save results to memeory in the results folder.
    The confusion matrices are as pdf files.
    Additional results are saved in .tex files.
    """

    if N==2:
        #compute metrics
        acc = accuracy_score(y_true, y_pred, normalize=True)
        se = recall_score(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sp = float(tn) / float((tn + fp))

        pr = precision_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_m = auc(fpr, tpr)

        # write to file
        with open(os.path.join('results', 'tab-'+fname+'.tex'), 'a+') as fout:
            fout.write('%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\\\hline\n' % (version, acc, se, sp, pr, auc_m))

    else:
        cm = confusion_matrix(y_true, y_pred)
   
        # save plots
        plt.figure(figsize=(N, N))
        cm, categories = order_matrix(cm, categories)
        plot_confusion_matrix(cm, classes=categories, normalize=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        img_name = fname+'-'+version+'.pdf'
        plt.savefig(os.path.join('results', img_name), format='pdf', bbox_inches='tight', dpi=600)

        # compute metrics
        to_print = []

        TP = np.diag(cm)
        FP = cm.sum(axis=0) -TP
        FN = [sum(r) for r in cm] - TP
        TN = cm.sum() - (FP + FN + TP)


        g_acc = sum(TP)/sum(sum(cm))
     
        for i, cl in enumerate(categories):

            pc_acc = zero_div(TP[i]+TN[i], TP[i]+TN[i]+FP[i]+FN[i])
            pc_se = zero_div(TP[i], TP[i]+FN[i])
            pc_sp = zero_div(TN[i], TN[i] + FP[i])
            pc_pr = zero_div(TP[i], TP[i]+FP[i])

            s = '%s & %.4f & %.4f & %.4f & %.4f \\\\\\hline' % (cl, pc_acc, pc_se, pc_sp, pc_pr)
            to_print.append(s)

        # write to file
        if rej is None:
            rej_str = ''
        else:
            rej_str = ', REJ = %.4f' % rej
        s = '\\begin{table}[htbp!]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|}\\hline\n    & PC-ACC & PC-SE & PC-SP & PC-PR \\\\\\hline\n'
        with open(os.path.join('results', 'tab-'+fname+'.tex'), 'a+') as fout:          
            fout.write(s)
            for p in to_print:
                fout.write(p+'\n')
            fout.write('\\end{tabular}\n\\caption{Results obtained with %s method. ACC = %.4f%s}\n\\end{table}\n\n\n' % (version, g_acc, rej_str))

        s = '\\begin{subfigure}[b]{0.32\\textwidth}\n\\centering\n\\includegraphics[width=\\textwidth]{imgs/%s}\n\\caption{%s.\\\\ACC=%.4f%s}\n\\end{subfigure}\n' % (img_name, version, g_acc, rej_str)
        with open(os.path.join('results', 'imgs-'+fname+'.tex'), 'a+') as fout:           
            fout.write(s)

def get_features(DCT, META, Header, m):
    """
    Return the correct set of features to use given the method m
    """
    if m == 'DCT':
        features = DCT     
    elif m == 'META':
        features = META
    elif m == 'DCT+META':
        features = np.concatenate((DCT, META), axis=1)
    elif m == 'HEADER':
        features = Header
    elif m == 'HEADER+META':
        features = np.concatenate((Header,  META), axis=1)
    elif m == 'HEADER+DCT+META':
        features = np.concatenate((Header, DCT,  META), axis=1)
    return features

def read_features(db, db_config, t):
    """
    Read features and label from h5py given
    the dataset, its configuration and the split t (train, test or validation)
    """
    hf = h5py.File('features/%s-conf%d.hdf5' % (db, db_config), 'r')

    DCT = np.asarray(hf[t+'/features/dct'])
    META = np.asarray(hf[t+'/features/meta'])
    Header = np.asarray(hf[t+'/features/header'])

    labels = np.asarray(hf[t+'/labels'])
    labels = [l.decode("utf-8") for l in labels]

    return DCT, META, Header, labels

def get_x_y(features, labels, map_lbls):
    """
    Labels conversion
    """
    x = []
    y = []
    for feat, lbl in zip(features, labels):
        if lbl in map_lbls:
            x.append(feat)
            y.append(map_lbls[lbl])
    return x, y

def get_predictor(name, DCT_train, META_train, Header_train, labels_train, map_lbls, features_to_use):
    """
    Returns a specific Random Forest predictor
    """
    pred = {}
     
    for m in features_to_use:  
        features = get_features(DCT_train, META_train, Header_train, m)
        x, y = get_x_y(features, labels_train, map_lbls)
        #x, y = balance(x, y)

        clf_rf = ensemble.RandomForestClassifier(n_estimators=10, random_state=1, class_weight='balanced')
        ovr_clf_rf = OneVsRestClassifier(clf_rf, n_jobs=-1)
        ovr_clf_rf.fit(x, y)
        pred[name+'-'+m] = ovr_clf_rf

    return pred

def get_predictors(DCT_train, META_train, Header_train, labels_train, classes, db_config, features_to_use):
    """
    Trains all the predictors and saves them to a dictionary.
    Another dictionary with the classes of each predictor is also populated and returned.
    """
    predictors = {}
    all_maps = {}

    # get predictor D1
    map_lbls = {}

    for lbl in classes[db_config-1]:
        map_lbls[lbl] = classes[0].index(lbl.split('-')[-1])
    predictors.update(get_predictor('D1', DCT_train, META_train, Header_train, labels_train, map_lbls, features_to_use))
    all_maps['D1'] = classes[0]
    
    # get D2 predictors
    if db_config>1:
        for cl in classes[0]:
            map_lbls = {}
            key = 'D2-'+cl
            new_cl = [lbl for lbl in classes[1] if lbl.endswith(cl)]
            for lbl in classes[db_config-1]:
                if lbl in new_cl:
                    map_lbls[lbl] = new_cl.index(lbl)
                else:
                    if lbl.count('-')==2:
                        for i, cl2 in enumerate(new_cl[1:]):
                            if lbl.endswith(cl2):
                                map_lbls[lbl] = i+1

            predictors.update(get_predictor(key, DCT_train, META_train, Header_train, labels_train, map_lbls, features_to_use))
            all_maps[key] = new_cl

    # get D3 preditcots
    if db_config==3:
        tmp_cl = [lbl for lbl in classes[1] if '-' in lbl]
        for cl in tmp_cl:
            map_lbls = {}
            key = 'D3-'+cl
            new_cl = [lbl for lbl in classes[2] if lbl.endswith(cl)]
            for i, lbl in enumerate(new_cl):
                map_lbls[lbl] = i

            predictors.update(get_predictor(key, DCT_train, META_train, Header_train, labels_train, map_lbls, features_to_use))
            all_maps[key] = new_cl
            
    return predictors, all_maps

def read_or_generate_predictors(db, classes, db_config, features_to_use, reduce_methods=False):
    """
    Train and return the predictors
    """
    DCT_train, META_train, Header_train, labels_train = read_features(db, db_config, 'train')
    if reduce_methods:
        tmp = []
        for f in features_to_use:
            tmp.extend(f.split('+'))
        features_to_use=list(set(tmp))

    predictors, maps = get_predictors(DCT_train, META_train, Header_train, labels_train, classes, db_config, features_to_use)
        
    return predictors, maps

def save_waterfall(N, cm, categories, fname, version, rej=None):
    """
    Save cascade results to memory.
    Confusion matrices are saved as pdf files.
    Additional results are seved in .tex files.
    """
    plt.figure(figsize=(N, N))
    cm, categories = order_matrix(cm, categories)
    np.savetxt(os.path.join('results', fname+'-'+version+'.csv'), cm, delimiter=",")
    plot_confusion_matrix(cm, classes=categories, normalize=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    img_name = fname+'-'+version+'.pdf'
    plt.savefig(os.path.join('results', img_name), format='pdf', bbox_inches='tight', dpi=600)

    # compute metrics
    to_print = []

    TP = np.diag(cm)
    FP = cm.sum(axis=0) -TP
    FN = [sum(r) for r in cm] - TP
    TN = cm.sum() - (FP + FN + TP)


    g_acc = sum(TP)/sum(sum(cm))
    
    for i, cl in enumerate(categories):

        pc_acc = zero_div(TP[i]+TN[i], TP[i]+TN[i]+FP[i]+FN[i])
        pc_se = zero_div(TP[i], TP[i]+FN[i])
        pc_sp = zero_div(TN[i], TN[i] + FP[i])
        pc_pr = zero_div(TP[i], TP[i]+FP[i])

        s = '%s & %.4f & %.4f & %.4f & %.4f \\\\\\hline' % (cl, pc_acc, pc_se, pc_sp, pc_pr)
        to_print.append(s)

    # write to file
    if rej is None:
        rej_str = ''
    else:
        rej_str = ', REJ = %.4f' % rej

    s = '\\begin{table}[htbp!]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|}\\hline\n    & PC-ACC & PC-SE & PC-SP & PC-PR \\\\\\hline\n'
    with open(os.path.join('results', 'tab-'+fname+'.tex'), 'a+') as fout:          
        fout.write(s)
        for p in to_print:
            fout.write(p+'\n')
        fout.write('\\end{tabular}\n\\caption{Results obtained with %s method. ACC = %.4f%s}\n\\end{table}\n\n\n' % (version, g_acc, rej_str))

    s = '\\begin{subfigure}[b]{0.32\\textwidth}\n\\centering\n\\includegraphics[width=\\textwidth]{imgs&tabs/%s}\n\\caption{%s. ACC=%.4f%s}\n\\end{subfigure}\n' % (img_name, version, g_acc, rej_str)
    with open(os.path.join('results', 'imgs-'+fname+'.tex'), 'a+') as fout:           
        fout.write(s)

def get_fuser(DCT_valid, META_valid, Header_valid, labels_valid, predictors, pred_name, map_lbls, methods_fusion, classes_names):
    """
    Train a specific BKS fuser
    """
    fuser = {}
    
    # Loop over fusion combinations (e.g., 'DCT+META')
    for m_fusion in methods_fusion:
        
        pred_outs = []
        # Loop over single predictors (e.g., ['DCT', 'META'])
        for m in m_fusion.split('+'):
        
            features = get_features(DCT_valid, META_valid, Header_valid, m)
            x, y = get_x_y(features, labels_valid, map_lbls)
            y = [classes_names[t] for t in y]

            # Get predictions
            tmp_p = predictors[pred_name+'-'+m].predict(x)
            pred_outs.append([classes_names[p] for p in tmp_p])
    
        fuser[m_fusion] = BKSFuser(len(m_fusion.split('+')), len(set(y)), classes_names)
        fuser[m_fusion].train(pred_outs, y)

    return fuser

def read_or_get_fusers(db, db_config, predictors, classes, methods_fusion, maps):
    """
    Train BKS fusers and return them
    """
    DCT_valid, META_valid, Header_valid, labels_valid = read_features(db, db_config, 'validation')
    fusers = {}

    # get predictor D1
    map_lbls = {}

    for lbl in classes[db_config-1]:
        map_lbls[lbl] = classes[0].index(lbl.split('-')[-1])

    fusers['D1'] = get_fuser(DCT_valid, META_valid, Header_valid, labels_valid, predictors, 'D1', map_lbls, methods_fusion, classes[0])
    
    # get D2 predictors
    if db_config>1:
        for cl in classes[0]:
            map_lbls = {}
            key = 'D2-'+cl
            new_cl = [lbl for lbl in classes[1] if lbl.endswith(cl)]
            for lbl in classes[db_config-1]:
                if lbl in new_cl:
                    map_lbls[lbl] = new_cl.index(lbl)
                else:
                    if lbl.count('-')==2:
                        for i, cl2 in enumerate(new_cl[1:]):
                            if lbl.endswith(cl2):
                                map_lbls[lbl] = i+1

            fusers[key] = get_fuser(DCT_valid, META_valid, Header_valid, labels_valid, predictors, key, map_lbls, methods_fusion, new_cl)

    # get D3 predictors
    if db_config==3:
        tmp_cl = [lbl for lbl in classes[1] if '-' in lbl]
        for cl in tmp_cl:
            map_lbls = {}
            key = 'D3-'+cl
            new_cl = [lbl for lbl in classes[2] if lbl.endswith(cl)]
            for i, lbl in enumerate(new_cl):
                map_lbls[lbl] = i

            fusers[key] = get_fuser(DCT_valid, META_valid, Header_valid, labels_valid, predictors, key, map_lbls, methods_fusion, new_cl)

    return fusers