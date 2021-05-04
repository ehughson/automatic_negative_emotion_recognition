from matplotlib.pyplot import xlabel, ylabel
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import svm
import random
from sklearn.utils import shuffle
import bisect



def create_svm(X_train, X_valid, y_train, y_valid):
	clf = svm.SVC(kernel ='linear',gamma=1.5, C=1.8) #linear kernel seems to work the best for North American Datatset, with gamma = 1.7 and C = 1.8; Philipines was gamma = 1.5; Persian was gamma = 1
	clf = clf.fit(X_train, y_train)
	print(clf.score(X_train, y_train))
	predictions = clf.predict(X_valid)
	print(classification_report(y_valid, predictions))
	print(accuracy_score(y_valid, predictions))
	score = accuracy_score(y_valid, predictions)
	fscore = f1_score(y_valid, predictions, average=None)
	return clf, score, fscore

def balance_data(df):
	df_c = df[df['emotion'] == 'contempt']
	df_a = df[df['emotion'] == 'anger']
	df_d = df[df['emotion'] == 'disgust']
	filename_contempt = df_c['filename'].unique()
	filename_anger = df_a['filename'].unique()
	filename_digust = df_d['filename'].unique()

	min_length = min(len(filename_contempt), len(filename_anger ), len(filename_digust))
	print("maximum length is: ", min_length)

	df_c_shuff = shuffle(filename_contempt)
	df_a_shuff = shuffle(filename_anger)
	df_d_shuff= shuffle(filename_digust)

	df_c = df_c[df_c["filename"].isin(df_c_shuff[0: min_length])]
	df_a = df_a[df_a["filename"].isin(df_a_shuff[0: min_length])]
	df_d = df_d[df_d["filename"].isin(df_d_shuff[0: min_length])]

	#print(len(df_c))
	#print(len(df_a))
	#print(len(df_d))

	df = pd.concat([df_c, df_a, df_d])
	return df


def k_fold_val(df, experiment_set, total_cf_matrix):
	kfold = KFold(5, True, 1)
	videos = df['filename'].unique()
	videos = np.array(list(set(videos)))
	splits = kfold.split(videos)
	le = LabelEncoder()
	validation_array = []
	test_array = []
	vf_score = []
	tf_score = []
	
	for (i, (train, test)) in enumerate(splits):
		print(test)
		print(train)
		print('%d-th split: train: %d, test: %d' % (i+1, len(videos[train]), len(videos[test])))
		
		train_df = df[df['filename'].isin(videos[train])]
		print(train_df.head())
		#test_df = df[df['filename'].isin(test_videos[test])]
		test_df = df[df['filename'].isin(videos[test])]
		y = train_df['emotion'].values
		X = train_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
		## Change labels to int using a label encoder
		Y = le.fit_transform(train_df['emotion'].values)
		#print(Y)
		X_train, X_valid, y_train, y_valid = train_test_split(X, Y)
		#print(X_train)
		print('LABEL ENCODER CLASSES: ', le.classes_)
		clf, score, fscore = create_svm(X_train, X_valid, y_train, y_valid)
		validation_array.append(score)
		vf_score.append(fscore)
		#cv_scores = cross_validate(clf, X, y, cv = 10)
		#print(cv_scores)
		# print(test_df[['frame','filename','culture','emotion']].head())

		int_test = test_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
		# print(len(int_test))
		## Roya: change string labels to integer values
		print(test_df['emotion'].unique())
		val = test_df['emotion'].values

		predictions = clf.predict(int_test) #integers predicted
		le_classes = le.classes_.tolist()
		#print(le_classes)
		#print(predictions)
		
		for i in predictions:
			if i not in le_classes:
				#print(i)
				#.insert(le_classes, i)
				le_classes.append(i)
		
		#bisect.insert_left(le_classes, '<unknown>')
		le.classes_ = le_classes
		int_predict = le.fit_transform(test_df['emotion'].values) 
		## Roya: change integer labels to string values
		test_df['predicted'] = le.inverse_transform(predictions)
		## Roya: calculate confusion matrix
		cf_matrix = confusion_matrix(test_df['emotion'].values, test_df['predicted'].values)


		print('CONFUSION MATRIX:\n', cf_matrix)
		print(cf_matrix/np.sum(cf_matrix))
		print(cf_matrix + cf_matrix)
		total_cf_matrix = total_cf_matrix + cf_matrix
		
		df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=le.inverse_transform([0,1,2]), columns=le.inverse_transform([0,1,2]))
		## Plot Confusion matrix
		df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
		plt.figure(figsize=(9,6))
		sn.heatmap(df_cm, annot=True,  fmt='.0%')
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		#plt.show()
		
		# print("predictions: ", predictions[0:10])
		# print("int_predict: ", int_predict[0:10])
		print(accuracy_score(int_predict, predictions))
		fscore = f1_score(le.fit_transform(int_predict), predictions, average = 'macro')
		test_df.drop(columns=['predicted'], inplace=True)
		print('\n')

		test_array.append(accuracy_score(int_predict, predictions))
		tf_score.append(fscore)
		


	y = train_df['emotion'].values
	Y = le.fit_transform(y)
	print(le.classes_)
	print("the confusion matrix:", total_cf_matrix/np.sum(total_cf_matrix))
	df_cm = pd.DataFrame(total_cf_matrix/np.sum(total_cf_matrix), index=le.inverse_transform([0,1,2]), columns=le.inverse_transform([0,1,2]))
	## Plot Confusion matrix
	df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
	plt.figure(figsize=(9,6))
	sn.heatmap(df_cm, annot=True,  fmt='.0%')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.title("Confusion Matrix for Within-Culture Testing on " + experiment_set +  " Dataset")
	#plt.show()
	if experiment_set != "North American":
		plt.savefig("Withinculture" + experiment_set)
	else:
		plt.savefig("WithincultureNA"
		)
	print("CURRENTLY TRAINING AND TESTING ON....." + experiment_set)

	print("Average accuracy for all Folds on valid dataset: " + str(np.mean(validation_array)))

	print("Average accuracy for all Folds on test dataset: " + str(np.mean(test_array)))

	print("Average f-score for all Folds on valid dataset: " + str(np.mean(vf_score)))

	print("Average f-score for all Folds on test dataset: " + str(np.mean(tf_score)))
	return test_array, np.mean(test_array), total_cf_matrix



def main():
	columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU23_r', 'AU25_r', 'AU26_r','culture','emotion']
	df = pd.read_csv("videos_relabelled.csv") 
	col = ['gender', 'talking','AU45_r']
	#col = ['gender', 'talking']
	df['gender'] = df['gender'].str.strip() #remove any spaces
	df['talking'] = df['talking'].str.strip() #remove any spaces
	df['emotion'] = df['emotion'].str.strip() #remove any spaces
	df = df[df['confidence'] >= 0.8]
	df = df[df['success'] != 0]
	df = df.drop(columns = col)

	df_1 = df[df['culture'] == 'Persian'] #training and testing on only persian culture
	df_2 = df[df['culture'] == 'Philippines'] #training and testing on only philipines culture
	df_3 = df[df['culture'] == 'North America'] #training and testing on only NA culture

	#df['culture_code'] = df['culture'].astype('category').cat.codes

	
	#print("dataframe length after balancing dataset: ", len(df1))
	den_p = 0
	den_f = 0
	den_n = 0
	double_average_p = []
	double_average_f = []
	double_average_n = []

	cf_p= [[0, 0, 0],[0, 0, 0], [0, 0, 0]]

	cf_n = [[0, 0, 0],[0, 0, 0], [0, 0, 0]]

	cf_f = [[0, 0, 0],[0, 0, 0], [0, 0, 0]]
	for i in range(0, 3):
		df1 = balance_data(df_1)
		df2 = balance_data(df_2)
		df3 = balance_data(df_3)
		valid_res_p, avg_p, cf_p = k_fold_val(df1, "Persian", cf_p)
		double_average_p.append(avg_p)

		valid_res_f, avg_f, cf_f = k_fold_val(df2, "Philippines",cf_f )
		double_average_f.append(avg_f)

		valid_res_n, avg_n, cf_n= k_fold_val(df3, "North American", cf_n)
		double_average_n.append(avg_n)

	#print(np.mean(array_test))
	print("the overall average for NA is:", str(np.mean(double_average_n)))

	print("the overall average for Persian is:", str(np.mean(double_average_p)))

	print("the overall average for Filipino is:", str(np.mean(double_average_f)))

if __name__=='__main__':
	#train_data = sys.argv[1]
	#test_data = sys.argv[2]
	main()