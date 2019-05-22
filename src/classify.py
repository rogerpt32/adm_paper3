import sys
import argparse
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

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


def get_classifier(method):
    method = method.lower()
    if method=="naive_bayes" or method=="naivebayes" or method=="nb":
        clf=GaussianNB()
    elif method=="decision_tree" or method=="decisiontree" or method=="dt":
        clf=DecisionTreeClassifier()
    elif method=="svm" or method=="supportvectormachine":
        clf=SVC()
    elif method=="random_forest" or method=="randomforest" or method=="rf":
        clf=RandomForestClassifier()
    elif method=="adaboost" or method=="ada_boost" or method=="ab":
        clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),n_estimators=100,random_state=123)
        # clf=AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=100,random_state=123)
    elif method=="mlp" or method=="neuralnetwork" or method=="multilayerperceptron" or method=="neural_network":
        # clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(520,300,200,180,120,60,90,60,12,10,10,6), random_state=123, early_stopping=False)
        clf = MLPClassifier(solver='adam', tol=1e-5, alpha=1e-5, hidden_layer_sizes=(560,560,280,140,140,70,70,70,35,30,30,18,12,10,6), 
                                n_iter_no_change=100, random_state=123, early_stopping=False,verbose=True)
    else:
        print("Error: undefined method: %s"%method)
        clf=None
    return clf

def plot_confusion_matrix(cm,test_y,pred_y,method,show,save):
    classes = list(set(test_y))
    classes.sort()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    parser.add_argument('--methods', help='delimited list input', type=str, default="naive_bayes,decision_tree,svm,mlp,random_forest,adaboost")
    parser.add_argument('--show_plot', action='store_true', help='show the confusion matrices with plots')
    parser.add_argument('--save_plot', action='store_true', help='save the confusion matrices with plots')

    args = parser.parse_args()

    methods = [item for item in args.methods.split(',')]
    # print(methods)
    data = pd.read_csv("../data/train.csv")
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