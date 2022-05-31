import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
def divide_columns_by_max(dataframe):
    newFrame = []
    for row in np.array(dataframe.values).T:
        newFrame.append([(item*50.0)/max(row)*1.0 for item in row])
    return np.array(newFrame).T
def accuracy_tests(TEST_NUMBER,test_train_subsets):
    models = [["SVM",SVC(kernel='poly',degree=12),0],["LR",LogisticRegression(max_iter=1000),0],["ID3",DecisionTreeClassifier(),0],["NB",GaussianNB(),0],["KNN",KNeighborsClassifier(),0]]
    # -----------Application of models on test samples---------------
    for model in models:
        for i in range(0,TEST_NUMBER):
            model[1] = model[1].fit(test_train_subsets[i][0],test_train_subsets[i][2])
            y_pred = model[1].predict(test_train_subsets[i][1])
            model[2] += (accuracy_score(test_train_subsets[i][3],y_pred)/TEST_NUMBER)
    for model in models:
        print('Accuracy of ',model[0],' = ',model[2])
def feature_selection_tests(dataset, number_of_features,test_train_subsets):
    Feature_Models = [["Linear SVM",SVC(kernel='linear'),0],["LR",LogisticRegression(max_iter=1000),0],["Decision Tree",DecisionTreeClassifier(),0]]
    for model in Feature_Models:
        for i in range(0,1):
            model[1] = RFE(model[1],n_features_to_select=number_of_features)
            model[1] = model[1].fit(test_train_subsets[i][0],test_train_subsets[i][2])
            model[2] += model[1].ranking_
    data_sets_feature_selected = []
    for feature_model in Feature_Models:
        x = ([i for i in range(len(feature_model[2])) if (feature_model[2][i]==1)])
        y = list(dataset.keys()[:-1][x])
        y.append('Loan_Status')
        #dataset = dataset[['LoanAmount','CoapplicantIncome','Property_Area','ApplicantIncome','Credit_History','Loan_Amount_Term','Loan_Status']]
        dt = dataset[y]
        data_sets_feature_selected.append(divide_columns_by_max(dt))

    models = [["SVM",SVC(kernel='poly',degree=12)],["LR",LogisticRegression(max_iter=1000)],["NB",GaussianNB()],["KNN",KNeighborsClassifier()]]
    test_data = []
    for feature_dataset in data_sets_feature_selected:
        arr = feature_dataset
        x = arr[:,0:-1]
        y = arr[:,-1]
        a = list(train_test_split(x,y,test_size=0.2))
        test_data.append(a)
    for testing_sub_dataset in test_data:
        for model in models:
            model[1] = model[1].fit(testing_sub_dataset[0],testing_sub_dataset[2])
            y_pred = model[1].predict(testing_sub_dataset[1])
            print(model[0]," accuracy: ", accuracy_score(testing_sub_dataset[3],y_pred))
        print("---------------------------")
