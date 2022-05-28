import pandas as pd
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
def featureSelectionPlot(TEST_NUMBER,test_train_subsets,dataset):
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 8}
    plt.rc('font', **font)
    Feature_Models = [["Linear SVM",SVC(kernel='linear'),0],["LR",LogisticRegression(max_iter=1000),0],["Decision Tree",DecisionTreeClassifier(),0]]
    for feature_model in Feature_Models:
        for i in range(0,TEST_NUMBER):
            selector = RFE(feature_model[1],n_features_to_select=4,step=1)
            selector = selector.fit(test_train_subsets[i][0],test_train_subsets[i][2])
            feature_model[2] += selector.ranking_
    for feature_model in Feature_Models:
        key_model_list = list(zip(dataset.keys()[:-1],feature_model[2]))
        key_model_list.sort(key=lambda row: row[1])
        keys, values = zip(*key_model_list)
        plt.bar(keys,height= values)
        plt.title(feature_model[0])
        plt.show()