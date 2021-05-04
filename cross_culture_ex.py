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



def train_dataframe(x, test_df, dataset_name1, dataset_name2):
    le = LabelEncoder()
    y = x['emotion'].values
    X = x.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
    Y = le.fit_transform(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y)
    clf, score, fscore = create_svm(X_train, X_valid, y_train, y_valid)
    print("the current dataset is: " + dataset_name1 + " and " + dataset_name2)
    print("validation score: " + str(score))
    print("validation f1score: " + str(fscore))
    #cv_scores = cross_validate(clf, X, y, cv = 10)
    #print(cv_scores)
    # print(test_df[['frame','filename','culture','emotion']].head())
    int_test = test_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
    # print(len(int_test))
    ## Roya: change string labels to integer values
    int_predict = le.fit_transform(test_df['emotion'].values) 
    # print(len(int_predict))
    # predictions = clf.predict(int_test)
    predictions = clf.predict(int_test) #integers predicted
    ## Roya: change integer labels to string values
    test_df['predicted'] = le.inverse_transform(predictions)
    ## Roya: calculate confusion matrix
    cf_matrix = confusion_matrix(test_df['emotion'].values, test_df['predicted'].values)
    print('CONFUSION MATRIX:\n', cf_matrix)
    print("accuracy on test: ", str(accuracy_score(int_predict, predictions)))
    fscore = f1_score(le.fit_transform(int_predict), predictions, average = 'macro')
    test_df.drop(columns=['predicted'], inplace=True)
    print("f1score on test: ", str(fscore))
    print('\n')
    return accuracy_score(int_predict, predictions)




def main():
    columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU23_r', 'AU25_r', 'AU26_r','culture','emotion']
    df = pd.read_csv("videos_relabelled.csv") 
    col = ['gender', 'talking','AU45_r']
    #col = ['gender', 'talking']
    df['gender'] = df['gender'].str.strip() #remove any spaces
    df['talking'] = df['talking'].str.strip() #remove any spaces
    df['emotion'] = df['emotion'].str.strip() #remove any spaces
    df = df.drop(columns = col)
    df = df[df['confidence'] >= 0.8]
    df = df[df['success'] != 0]
    df1 = df[df['culture'] == 'Persian'] #training and testing on only persian culture
    df2 = df[df['culture'] == 'Philippines'] #training and testing on only philipines culture
    df3 = df[df['culture'] == 'North America'] #training and testing on only NA culture

    #df['culture_code'] = df['culture'].astype('category').cat.codes

    double_average_p_f = []
    double_average_f_p = []
    double_average_n_f = []
    double_average_f_n = []
    double_average_n_p = []
    double_average_p_n = []

    for i in range(0, 3):
        df1 = balance_data(df1)
        df2 = balance_data(df2)
        df3 = balance_data(df3)

        p_f_test_acc = train_dataframe(df1, df2,"Persian", "Philippines" ) #training on Persian, testing Filipino
        double_average_p_f.append(p_f_test_acc)


        f_p_test_acc = train_dataframe(df2, df1,"Philippines", "Persian" ) #training on Persian, testing Filipino
        double_average_f_p.append(f_p_test_acc)

        n_f_test_acc = train_dataframe(df3, df2,"NA", "Philippines" ) #training on Persian, testing Filipino
        double_average_n_f.append(n_f_test_acc)

        f_n_test_acc = train_dataframe(df2, df3,"Philippines", "NA" ) #training on Persian, testing Filipino
        double_average_f_n.append(f_n_test_acc)

        n_p_test_acc = train_dataframe(df3, df1,"NA", "Persian" ) #training on Persian, testing Filipino
        double_average_n_p.append(n_p_test_acc)


        p_n_test_acc = train_dataframe(df1, df3,"Persian", "NA" ) #training on Persian, testing Filipino
        double_average_p_n.append(p_n_test_acc)
    

    print("the overall average for training on persian and testing on filipino is:", str(np.mean(double_average_p_f)))

    print("the overall average for training on filipino and testing on persian is:", str(np.mean(double_average_f_p)))

    print("the overall average for training on NA and testing on Filipino is:",str(np.mean(double_average_n_f)))

    print("the overall average for training on filipino and testing on NA is:", str(np.mean(double_average_f_n)))

    print("the overall average for training on NA and testing on persian is:", str(np.mean(double_average_n_p)))

    print("the overall average for training on Persian and testing on NA is:", str(np.mean(double_average_p_n)))
    
    
    
    
    
    
    return


if __name__=='__main__':
	#train_data = sys.argv[1]
	#test_data = sys.argv[2]
	main()