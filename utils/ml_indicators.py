import numpy as np 


'''
Regression
1. MSE
2. MAE
3. RMSE
4. MPE
5. MAPE
6. RMSLE
'''

def MSE_score(y_test, pred):
    return np.square(np.subtract(y_test, pred)).mean()

def MAE_score(y_test, pred):
    return np.abs(np.subtract(y_test, pred)).mean()

def RMSE_score(y_test, pred):
    return np.sqrt(np.square(np.subtract(y_test, pred))).mean()

def MPE_score(y_test, y_pred): 
	return np.mean((y_test - y_pred) / y_test) * 100

def MAPE_score(y_test, y_pred): 
	return np.mean(np.abs(y_test - y_pred) / y_test) * 100

def RMSLE_score(y_test, y_pred): 
    return np.sqrt(np.square(np.log(y_test + 1) - np.log(y_pred + 1)).mean())


'''
Classification
1. Accuracy
2. recall
3. precision
4. f1 
'''
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def accuracy_score(y_test, pred):
    return sum(np.equal(y_test, pred))/ len(y_test)

def recall_score_macro(target, pred, average='macro'):
    return recall_score(target, pred, average=average)

def precision_score_macro(target, pred, average='macro'):
    return precision_score(target, pred, average=average)

def f1_score_macro(target, pred, average='macro'):
    return f1_score(target, pred, average=average)