import pandas as pd
from feature_selection_plots import featureSelectionPlot
from sklearn.model_selection import train_test_split
from accuracy_test import accuracy_tests,feature_selection_tests
TEST_NUMBER = 1     #number of tests to be performed for evaluation
# get csv file from path
dataset = pd.read_csv('loan_data.csv')
dataset = dataset.iloc[:,1:] #remove ID column
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
dataset = dataset.apply(lambda x: pd.factorize(x)[0])
arr = dataset.values
x = arr[:,0:-1]
y = arr[:,-1]
test_train_subsets = []
# -----------Test sampling---------------
for i in range(0,TEST_NUMBER):
    a , b ,c ,d = train_test_split(x,y,test_size=0.2,random_state=i)
    test_train_subsets.append((a, b ,c ,d))
## run accuracy tests
# accuracy_tests(TEST_NUMBER,test_train_subsets)
## --------Feature Selection plotting-----------
# featureSelectionPlot(TEST_NUMBER,test_train_subsets,dataset)
# feature_selection_tests(dataset,9,test_train_subsets)