from sklearn import svm
import pandas as pd
import numpy as np
import sys
from operator import itemgetter
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_validate


def create_svm(X_train, X_valid, y_train, y_valid):
	#scaler = StandardScaler().fit(X_train)
	#X_scaled = scaler.transform(X_train)
	#X_valid_scaled = scaler.transform(X_valid)
	clf = svm.SVC(kernel ='rbf',gamma=1.5, C=2.8)
	#clf = svm.SVC(kernel='linear')

	#clf = AdaBoostClassifier(clf, learning_rate=0.6,  algorithm='SAMME')
	#cv_scores = cross_validate(clf, X, y, cv = 10, scoring = ['recall', 'precision','accuracy'])
	#print(cv_scores)
	clf = clf.fit(X_train, y_train)
	print(clf.score(X_train, y_train))
	print(clf.score(X_valid, y_valid))
	predictions = clf.predict(X_valid)
	print(classification_report(y_valid, predictions))
	print(accuracy_score(y_valid, predictions))
	return clf

def separate_emotions(X):


	print( '############# PHILIPPINES############### \n')
	#################### PHILIPPINES ###################
	culture_1 = X[(X['culture'] == 'Persian') | (X['culture'] == 'North America')]
	test = X[X['culture'] == 'Philippines']
	culture_1['culture_code'] = culture_1['culture'].astype('category').cat.codes
	y = culture_1['emotion'].values
	culture_1 = culture_1.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	
	X_train, X_valid, y_train, y_valid = train_test_split(culture_1, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)

	test['culture_code'] = test['culture'].astype('category').cat.codes
	int_test = test.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(len(int_test))
	int_predict = test['emotion'].values
	print(len(int_predict))
	predictions = clf.predict(int_test)
	print(accuracy_score(int_predict, predictions))
	print('\n')


	print( '############# NORTH AMERICA ############### \n')
	#################### NORTH AMERICA ###################
	culture_1 = X[(X['culture'] == 'Persian') | (X['culture'] == 'Philippines')]
	test = X[X['culture'] == 'North America']
	culture_1['culture_code'] = culture_1['culture'].astype('category').cat.codes
	y = culture_1['emotion'].values
	culture_1 = culture_1.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	
	X_train, X_valid, y_train, y_valid = train_test_split(culture_1, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)

	test['culture_code'] = test['culture'].astype('category').cat.codes
	int_test = test.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(len(int_test))
	int_predict = test['emotion'].values
	print(len(int_predict))
	predictions = clf.predict(int_test)
	print(accuracy_score(int_predict, predictions))
	print('\n')
	print( '############# PERSIAN ############### \n')
	#################### NORTH AMERICA ###################
	culture_1 = X[(X['culture'] == 'North America') | (X['culture'] == 'Philippines')]
	test = X[X['culture'] == 'Persian']
	culture_1['culture_code'] = culture_1['culture'].astype('category').cat.codes
	y = culture_1['emotion'].values
	culture_1 = culture_1.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	
	X_train, X_valid, y_train, y_valid = train_test_split(culture_1, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)

	test['culture_code'] = test['culture'].astype('category').cat.codes
	int_test = test.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(len(int_test))
	int_predict = test['emotion'].values
	print(len(int_predict))
	predictions = clf.predict(int_test)
	print(accuracy_score(int_predict, predictions))
	print('\n')




def main():
	
	columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','culture','emotion']
	X = pd.read_csv("all_videos.csv")
	separate_emotions(X)
	print(X.head())
	X['culture_code'] = X['culture'].astype('category').cat.codes
	y = X['emotion'].values
	X2 = X.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(X2[:10])

	X_train, X_valid, y_train, y_valid = train_test_split(X2, y)
	

	#create_classifier(X_train, X_valid, y_train, y_valid)
	clf = create_svm(X_train, X_valid, y_train, y_valid)
	

	

if __name__=='__main__':
	#train_data = sys.argv[1]
	#test_data = sys.argv[2]
	main()

