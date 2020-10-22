import pandas as pd
import researchpy as rp
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def bellCurve(df, au, col):
	labels = ["North America", "Persian", "Philippines"]
	culture1 = df.loc[df['culture'] == 'North America']
	culture2 = df.loc[df['culture'] == 'Persian']
	culture3 = df.loc[df['culture'] == 'Philippines']
	plt.hist([culture1[au], culture2[au], culture3[au]])
	plt.legend(labels)
	plt.xlabel("AU intensity")
	plt.ylabel("amount")
	plt.title("Initial Distributon of " + col)
	plt.savefig(r".\plots\distribution_1.png")

	plt.show()

def normalize(df, name, col):
	df = df[[col, 'emotion', 'culture', 'filename']]
	df[name] = df.groupby(['filename', 'culture'])[col].transform(max)
	#print(df.head())
	df = df.drop(columns = col)
	df = df.drop_duplicates(['filename'])[['culture',name]]
	#print(df.head())

	print(rp.summary_cont(df[name]))
	print(rp.summary_cont(df[name].groupby(df['culture'])))
	bellCurve(df, name, col)

def ANOVA(df, col):
	print("DESCRIPTIVE STATISTICS ON" +  col + "ACROSS CULTURES on CONTEMPT")
	df = df[[col, 'emotion', 'culture']]
	#print(df.head())

	oneway_anova = stats.f_oneway(df[col][df['culture'] == 'North America'],
		df[col][df['culture'] == 'Persian'],
		df[col][df['culture'] == 'Philippines'])
	print(oneway_anova)


def stat_analysis():
	columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','culture','emotion']
	df = pd.read_csv("videos_relabelled.csv")

	df_c = df[df['emotion'] == 'contempt']

	#to normalize data by getting max activation of a given AU and then visualize with a histogram
	normalize(df_c, "max_au4", 'AU04_r')
	normalize(df_c, "max_au7", 'AU07_r')
	normalize(df_c, "max_au10", 'AU10_r')
	normalize(df_c, "max_au25", 'AU25_r')
	normalize(df_c, "max_au26", 'AU26_r')
	

	#to get one-way ANOVA results
	ANOVA(df_c, 'AU04_r')
	ANOVA(df_c, 'AU07_r')
	ANOVA(df_c, 'AU10_r')
	ANOVA(df_c, 'AU25_r')
	ANOVA(df_c, 'AU26_r')



if __name__ == '__main__':
	 stat_analysis()




