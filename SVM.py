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
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate


def create_svm(X_train, X_valid, y_train, y_valid):
	#scaler = StandardScaler().fit(X_train)
	#X_scaled = scaler.transform(X_train)
	#X_valid_scaled = scaler.transform(X_valid)
	clf = svm.SVC(kernel ='rbf',gamma=1.5, C=1.8)
	#clf = svm.SVC(kernel='linear')

	#clf = AdaBoostClassifier(clf, learning_rate=0.6,  algorithm='SAMME')
	#cv_scores = cross_validate(clf, X, y, cv = 10, scoring = ['recall', 'precision','accuracy'])
	

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
	df = pd.read_csv("all_videos.csv")
	

	'''
	############# initial process on random sampling ####################
	#print(df.head())
	df['culture_code'] = df['culture'].astype('category').cat.codes
	y = df['emotion'].values
	X = df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	#print(X[:10])

	X_train, X_valid, y_train, y_valid = train_test_split(X, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)
	

	############# testing model by separating training on 2 cultures and test on 1 culture####################
	separate_emotions(df)
	'''

	############# testing model by selecting specific videos to test so components of video are not in training set ###############
	validation_array = []
	kfold = KFold(5, True, 1)
	videos = df['filename'].unique()
	test_videos = pd.Series(videos).sample(frac=0.10)
	# print(test_videos)
	# videos must be array to be subscriptable by a list
	videos = np.array(list(set(videos) - set(test_videos)))
	# Removing test videos from train dataset
	test_df = df[df['filename'].isin(test_videos)]
	df = df[~df['filename'].isin(list(test_videos))]
	splits = kfold.split(videos)
	test_df_copy = test_df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1)
	for (i, (train, test)) in enumerate(splits):
	    print('%d-th split: train: %d, test: %d' % (i+1, len(videos[train]), len(videos[test])))
	    train_df = df[df['filename'].isin(videos[train])]
	    test_df = df[df['filename'].isin(videos[test])]
	    y = train_df['emotion'].values
	    X = train_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values

	    X_train, X_valid, y_train, y_valid = train_test_split(X, y)


	    clf = create_svm(X_train, X_valid, y_train, y_valid)
	    #cv_scores = cross_validate(clf, X, y, cv = 10)
	    #print(cv_scores)

	    int_test = test_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	    print(len(int_test))
	    int_predict = test_df['emotion'].values
	    print(len(int_predict))
	    predictions = clf.predict(int_test)
	    print(accuracy_score(int_predict, predictions))
	    print('\n')

	    validation_array.append(accuracy_score(int_predict, predictions))
 
	print("Average accuracy for all Folds on test dataset: " + str(np.mean(validation_array )))


if __name__=='__main__':
	#train_data = sys.argv[1]
	#test_data = sys.argv[2]
	main()


