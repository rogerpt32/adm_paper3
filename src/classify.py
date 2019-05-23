import sys
import argparse
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['font.sans-serif'] = ['Console Modern']
rcParams['savefig.format'] = ['pdf']
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0


def get_standard_name(method):
    method = method.lower()
    if method=="naive_bayes" or method=="naivebayes" or method=="nb":
        name='naive_bayes'
    elif method=="decision_tree" or method=="decisiontree" or method=="dt":
        name='decision_tree'
    elif method=="svm" or method=="supportvectormachine":
        name='svm'
    elif method=="mlp" or method=="neuralnetwork" or method=="multilayerperceptron" or method=="neural_network":
        name='mlp'
    elif method=="random_forest" or method=="randomforest" or method=="rf":
        name='random_forest'
    elif method=="adaboost" or method=="ada_boost" or method=="ab":
        name='adaboost'
    else:
        name='error'
    return name

def get_classifier(method):
    method = method.lower()
    if method=="naive_bayes":
        clf=GaussianNB()

    elif method=="decision_tree":
        clf=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=123)

    elif method=="svm":
        # clf=SVC(kernel='rbf',C=0.9,random_state=123)
        # clf=SVC(kernel='poly',C=1.0,degree=2,random_state=123)
        # clf=SVC(kernel='sigmoid',C=10.0,coef0=0.1,tol=1e-5,random_state=123)
        clf=SVC(kernel='linear',C=1.0,random_state=123)

    elif method=="mlp":
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(12,6),
                                # verbose=True, # Uncomment this line to see how the mlp learns
                                n_iter_no_change=20, max_iter=500, early_stopping=False, tol=1e-3, alpha=1e-5, random_state=123)

    elif method=="random_forest":
        clf=RandomForestClassifier(max_depth=4,criterion='gini',random_state=123)

    elif method=="adaboost":
        clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=50,random_state=123)
        # clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),n_estimators=100,random_state=123)
        # clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,random_state=123),n_estimators=500,random_state=123)
        # clf=AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=100,random_state=123)
        # clf=AdaBoostClassifier(base_estimator=SVC(kernel='linear',random_state=123),n_estimators=10,algorithm='SAMME',random_state=123)

    else:
        print("Error: undefined method: %s"%method)
        clf=None

    return clf

def plot_confusion_matrix(cm,test_y,pred_y,method,show,save):
    classes = list(set(test_y))
    classes.sort()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=method,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig("../plots/cm_%s.pdf"%method)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description='Classify the data.')
    parser.add_argument('-m','--methods', help='delimited list input', type=str, default="naive_bayes,decision_tree,svm,mlp,random_forest,adaboost")
    parser.add_argument('-p','--show_plot', action='store_true', help='show the confusion matrices with plots')
    parser.add_argument('-s','--save_plot', action='store_true', help='save the confusion matrices with plots')

    args = parser.parse_args()

    methods = [get_standard_name(item) for item in args.methods.split(',')]
    # print(methods)
    data = pd.read_csv("../data/train.csv")
    data = data.drop("rn",axis=1)
    data = shuffle(data,random_state=123)
    train_x,test_x,train_y,test_y = train_test_split(data.drop("activity",axis=1),data[['activity']],train_size=0.75,random_state=123)
    train_y = train_y.values.ravel()
    test_y = test_y.values.ravel()

    print("Test values:\n")
    print("\tLAYING: %d"%list(test_y).count("LAYING"))
    print("\tSITTING: %d"%list(test_y).count("SITTING"))
    print("\tSTANDING: %d"%list(test_y).count("STANDING"))
    print("\tWALKING: %d"%list(test_y).count("WALKING"))
    print("\tWALKING_DOWNSTAIRS: %d"%list(test_y).count("WALKING_DOWNSTAIRS"))
    print("\tWALKING_UPSTAIRS: %d\n"%list(test_y).count("WALKING_UPSTAIRS"))
    
    print("#"*50)
    for method in methods:
        print("="*50)
        clf=get_classifier(method)
        if clf is not None:
            print("METHOD: %s \n"%method.lower())
            pred_y = clf.fit(train_x,train_y).predict(test_x)
            hits = 0
            miss = 0
            for i, j in zip(test_y, pred_y):
                if i==j:
                    hits+=1
                else:
                    miss+=1
            print("Hits = %d (%.2f %%)"%(hits,100*hits/len(test_y)))
            print("Miss = %d (%.2f %%)"%(miss,100*miss/len(test_y)))
            print("Total = %d"%len(test_y))
            print("Confusion Matrix: ")
            cm=confusion_matrix(test_y,pred_y)
            print(cm)
            if args.show_plot or args.save_plot:
                plot_confusion_matrix(cm,test_y,pred_y,method,args.show_plot,args.save_plot)
    print("="*50)
    print("#"*50)