### This code computes the mean and standard deviation of AUs in the dataset all_videos.csv ###
### Please ensure that all_videos.csv is located in the same directory as this file ###

import pandas as pd
import matplotlib.pyplot as plt
import statistics as st




df = pd.read_csv("all_videos.csv")

persianContemptAU = dict()
persianAngerAU = dict()
persianDisgustAU = dict()

naContemptAU = dict()
naAngerAU = dict()
naDisgustAU = dict()

philContemptAU = dict()
philAngerAU = dict()
philDisgustAU = dict()

#Initialize dict keys for persian culture
persianContemptAU['AU01_r'] = list()
persianContemptAU['AU02_r'] = list()
persianContemptAU['AU04_r'] = list()
persianContemptAU['AU05_r'] = list()
persianContemptAU['AU06_r'] = list()
persianContemptAU['AU07_r'] = list()
persianContemptAU['AU09_r'] = list()
persianContemptAU['AU10_r'] = list()
persianContemptAU['AU12_r'] = list()
persianContemptAU['AU14_r'] = list()
persianContemptAU['AU15_r'] = list()
persianContemptAU['AU17_r'] = list()
persianContemptAU['AU23_r'] = list()
persianContemptAU['AU25_r'] = list()
persianContemptAU['AU26_r'] = list()
persianContemptAU['AU45_r'] = list()

persianAngerAU['AU01_r'] = list()
persianAngerAU['AU02_r'] = list()
persianAngerAU['AU04_r'] = list()
persianAngerAU['AU05_r'] = list()
persianAngerAU['AU06_r'] = list()
persianAngerAU['AU07_r'] = list()
persianAngerAU['AU09_r'] = list()
persianAngerAU['AU10_r'] = list()
persianAngerAU['AU12_r'] = list()
persianAngerAU['AU14_r'] = list()
persianAngerAU['AU15_r'] = list()
persianAngerAU['AU17_r'] = list()
persianAngerAU['AU23_r'] = list()
persianAngerAU['AU25_r'] = list()
persianAngerAU['AU26_r'] = list()
persianAngerAU['AU45_r'] = list()


persianDisgustAU['AU01_r'] = list()
persianDisgustAU['AU02_r'] = list()
persianDisgustAU['AU04_r'] = list()
persianDisgustAU['AU05_r'] = list()
persianDisgustAU['AU06_r'] = list()
persianDisgustAU['AU07_r'] = list()
persianDisgustAU['AU09_r'] = list()
persianDisgustAU['AU10_r'] = list()
persianDisgustAU['AU12_r'] = list()
persianDisgustAU['AU14_r'] = list()
persianDisgustAU['AU15_r'] = list()
persianDisgustAU['AU17_r'] = list()
persianDisgustAU['AU23_r'] = list()
persianDisgustAU['AU25_r'] = list()
persianDisgustAU['AU26_r'] = list()
persianDisgustAU['AU45_r'] = list()

#Initialize dict keys for na culture

naContemptAU['AU01_r'] = list()
naContemptAU['AU02_r'] = list()
naContemptAU['AU04_r'] = list()
naContemptAU['AU05_r'] = list()
naContemptAU['AU06_r'] = list()
naContemptAU['AU07_r'] = list()
naContemptAU['AU09_r'] = list()
naContemptAU['AU10_r'] = list()
naContemptAU['AU12_r'] = list()
naContemptAU['AU14_r'] = list()
naContemptAU['AU15_r'] = list()
naContemptAU['AU17_r'] = list()
naContemptAU['AU23_r'] = list()
naContemptAU['AU25_r'] = list()
naContemptAU['AU26_r'] = list()
naContemptAU['AU45_r'] = list()

naAngerAU['AU01_r'] = list()
naAngerAU['AU02_r'] = list()
naAngerAU['AU04_r'] = list()
naAngerAU['AU05_r'] = list()
naAngerAU['AU06_r'] = list()
naAngerAU['AU07_r'] = list()
naAngerAU['AU09_r'] = list()
naAngerAU['AU10_r'] = list()
naAngerAU['AU12_r'] = list()
naAngerAU['AU14_r'] = list()
naAngerAU['AU15_r'] = list()
naAngerAU['AU17_r'] = list()
naAngerAU['AU23_r'] = list()
naAngerAU['AU25_r'] = list()
naAngerAU['AU26_r'] = list()
naAngerAU['AU45_r'] = list()


naDisgustAU['AU01_r'] = list()
naDisgustAU['AU02_r'] = list()
naDisgustAU['AU04_r'] = list()
naDisgustAU['AU05_r'] = list()
naDisgustAU['AU06_r'] = list()
naDisgustAU['AU07_r'] = list()
naDisgustAU['AU09_r'] = list()
naDisgustAU['AU10_r'] = list()
naDisgustAU['AU12_r'] = list()
naDisgustAU['AU14_r'] = list()
naDisgustAU['AU15_r'] = list()
naDisgustAU['AU17_r'] = list()
naDisgustAU['AU23_r'] = list()
naDisgustAU['AU25_r'] = list()
naDisgustAU['AU26_r'] = list()
naDisgustAU['AU45_r'] = list()

#Initialize dict key values for philippines culture

philContemptAU['AU01_r'] = list()
philContemptAU['AU02_r'] = list()
philContemptAU['AU04_r'] = list()
philContemptAU['AU05_r'] = list()
philContemptAU['AU06_r'] = list()
philContemptAU['AU07_r'] = list()
philContemptAU['AU09_r'] = list()
philContemptAU['AU10_r'] = list()
philContemptAU['AU12_r'] = list()
philContemptAU['AU14_r'] = list()
philContemptAU['AU15_r'] = list()
philContemptAU['AU17_r'] = list()
philContemptAU['AU23_r'] = list()
philContemptAU['AU25_r'] = list()
philContemptAU['AU26_r'] = list()
philContemptAU['AU45_r'] = list()

philAngerAU['AU01_r'] = list()
philAngerAU['AU02_r'] = list()
philAngerAU['AU04_r'] = list()
philAngerAU['AU05_r'] = list()
philAngerAU['AU06_r'] = list()
philAngerAU['AU07_r'] = list()
philAngerAU['AU09_r'] = list()
philAngerAU['AU10_r'] = list()
philAngerAU['AU12_r'] = list()
philAngerAU['AU14_r'] = list()
philAngerAU['AU15_r'] = list()
philAngerAU['AU17_r'] = list()
philAngerAU['AU23_r'] = list()
philAngerAU['AU25_r'] = list()
philAngerAU['AU26_r'] = list()
philAngerAU['AU45_r'] = list()


philDisgustAU['AU01_r'] = list()
philDisgustAU['AU02_r'] = list()
philDisgustAU['AU04_r'] = list()
philDisgustAU['AU05_r'] = list()
philDisgustAU['AU06_r'] = list()
philDisgustAU['AU07_r'] = list()
philDisgustAU['AU09_r'] = list()
philDisgustAU['AU10_r'] = list()
philDisgustAU['AU12_r'] = list()
philDisgustAU['AU14_r'] = list()
philDisgustAU['AU15_r'] = list()
philDisgustAU['AU17_r'] = list()
philDisgustAU['AU23_r'] = list()
philDisgustAU['AU25_r'] = list()
philDisgustAU['AU26_r'] = list()
philDisgustAU['AU45_r'] = list()





for index, row in df.iterrows():
    if row['culture'] == 'Persian' and row['emotion'] == 'contempt':
        persianContemptAU['AU01_r'].append(row['AU01_r'])
        persianContemptAU['AU02_r'].append(row['AU02_r'])
        persianContemptAU['AU04_r'].append(row['AU04_r'])
        persianContemptAU['AU05_r'].append(row['AU05_r'])
        persianContemptAU['AU06_r'].append(row['AU06_r'])
        persianContemptAU['AU07_r'].append(row['AU07_r'])
        persianContemptAU['AU09_r'].append(row['AU09_r'])
        persianContemptAU['AU10_r'].append(row['AU10_r'])
        persianContemptAU['AU12_r'].append(row['AU12_r'])
        persianContemptAU['AU14_r'].append(row['AU14_r'])
        persianContemptAU['AU15_r'].append(row['AU15_r'])
        persianContemptAU['AU17_r'].append(row['AU17_r'])
        persianContemptAU['AU23_r'].append(row['AU23_r'])
        persianContemptAU['AU25_r'].append(row['AU25_r'])
        persianContemptAU['AU26_r'].append(row['AU26_r'])
        persianContemptAU['AU45_r'].append(row['AU45_r'])
    elif row['culture'] == 'Persian' and row['emotion'] == 'anger':
        persianAngerAU['AU01_r'].append(row['AU01_r'])
        persianAngerAU['AU02_r'].append(row['AU02_r'])
        persianAngerAU['AU04_r'].append(row['AU04_r'])
        persianAngerAU['AU05_r'].append(row['AU05_r'])
        persianAngerAU['AU06_r'].append(row['AU06_r'])
        persianAngerAU['AU07_r'].append(row['AU07_r'])
        persianAngerAU['AU09_r'].append(row['AU09_r'])
        persianAngerAU['AU10_r'].append(row['AU10_r'])
        persianAngerAU['AU12_r'].append(row['AU12_r'])
        persianAngerAU['AU14_r'].append(row['AU14_r'])
        persianAngerAU['AU15_r'].append(row['AU15_r'])
        persianAngerAU['AU17_r'].append(row['AU17_r'])
        persianAngerAU['AU23_r'].append(row['AU23_r'])
        persianAngerAU['AU25_r'].append(row['AU25_r'])
        persianAngerAU['AU26_r'].append(row['AU26_r'])
        persianAngerAU['AU45_r'].append(row['AU45_r'])
    elif row['culture'] == 'Persian' and row['emotion'] == 'disgust':
        persianDisgustAU['AU01_r'].append(row['AU01_r'])
        persianDisgustAU['AU02_r'].append(row['AU02_r'])
        persianDisgustAU['AU04_r'].append(row['AU04_r'])
        persianDisgustAU['AU05_r'].append(row['AU05_r'])
        persianDisgustAU['AU06_r'].append(row['AU06_r'])
        persianDisgustAU['AU07_r'].append(row['AU07_r'])
        persianDisgustAU['AU09_r'].append(row['AU09_r'])
        persianDisgustAU['AU10_r'].append(row['AU10_r'])
        persianDisgustAU['AU12_r'].append(row['AU12_r'])
        persianDisgustAU['AU14_r'].append(row['AU14_r'])
        persianDisgustAU['AU15_r'].append(row['AU15_r'])
        persianDisgustAU['AU17_r'].append(row['AU17_r'])
        persianDisgustAU['AU23_r'].append(row['AU23_r'])
        persianDisgustAU['AU25_r'].append(row['AU25_r'])
        persianDisgustAU['AU26_r'].append(row['AU26_r'])
        persianDisgustAU['AU45_r'].append(row['AU45_r'])
        
    if row['culture'] == 'North America' and row['emotion'] == 'contempt':
        naContemptAU['AU01_r'].append(row['AU01_r'])
        naContemptAU['AU02_r'].append(row['AU02_r'])
        naContemptAU['AU04_r'].append(row['AU04_r'])
        naContemptAU['AU05_r'].append(row['AU05_r'])
        naContemptAU['AU06_r'].append(row['AU06_r'])
        naContemptAU['AU07_r'].append(row['AU07_r'])
        naContemptAU['AU09_r'].append(row['AU09_r'])
        naContemptAU['AU10_r'].append(row['AU10_r'])
        naContemptAU['AU12_r'].append(row['AU12_r'])
        naContemptAU['AU14_r'].append(row['AU14_r'])
        naContemptAU['AU15_r'].append(row['AU15_r'])
        naContemptAU['AU17_r'].append(row['AU17_r'])
        naContemptAU['AU23_r'].append(row['AU23_r'])
        naContemptAU['AU25_r'].append(row['AU25_r'])
        naContemptAU['AU26_r'].append(row['AU26_r'])
        naContemptAU['AU45_r'].append(row['AU45_r'])
    elif row['culture'] == 'North America' and row['emotion'] == 'anger':
        naAngerAU['AU01_r'].append(row['AU01_r'])
        naAngerAU['AU02_r'].append(row['AU02_r'])
        naAngerAU['AU04_r'].append(row['AU04_r'])
        naAngerAU['AU05_r'].append(row['AU05_r'])
        naAngerAU['AU06_r'].append(row['AU06_r'])
        naAngerAU['AU07_r'].append(row['AU07_r'])
        naAngerAU['AU09_r'].append(row['AU09_r'])
        naAngerAU['AU10_r'].append(row['AU10_r'])
        naAngerAU['AU12_r'].append(row['AU12_r'])
        naAngerAU['AU14_r'].append(row['AU14_r'])
        naAngerAU['AU15_r'].append(row['AU15_r'])
        naAngerAU['AU17_r'].append(row['AU17_r'])
        naAngerAU['AU23_r'].append(row['AU23_r'])
        naAngerAU['AU25_r'].append(row['AU25_r'])
        naAngerAU['AU26_r'].append(row['AU26_r'])
        naAngerAU['AU45_r'].append(row['AU45_r'])
    elif row['culture'] == 'North America' and row['emotion'] == 'disgust':
        naDisgustAU['AU01_r'].append(row['AU01_r'])
        naDisgustAU['AU02_r'].append(row['AU02_r'])
        naDisgustAU['AU04_r'].append(row['AU04_r'])
        naDisgustAU['AU05_r'].append(row['AU05_r'])
        naDisgustAU['AU06_r'].append(row['AU06_r'])
        naDisgustAU['AU07_r'].append(row['AU07_r'])
        naDisgustAU['AU09_r'].append(row['AU09_r'])
        naDisgustAU['AU10_r'].append(row['AU10_r'])
        naDisgustAU['AU12_r'].append(row['AU12_r'])
        naDisgustAU['AU14_r'].append(row['AU14_r'])
        naDisgustAU['AU15_r'].append(row['AU15_r'])
        naDisgustAU['AU17_r'].append(row['AU17_r'])
        naDisgustAU['AU23_r'].append(row['AU23_r'])
        naDisgustAU['AU25_r'].append(row['AU25_r'])
        naDisgustAU['AU26_r'].append(row['AU26_r'])
        naDisgustAU['AU45_r'].append(row['AU45_r'])
        
    if row['culture'] == 'Philippines' and row['emotion'] == 'contempt':   
        philContemptAU['AU01_r'].append(row['AU01_r'])
        philContemptAU['AU02_r'].append(row['AU02_r'])
        philContemptAU['AU04_r'].append(row['AU04_r'])
        philContemptAU['AU05_r'].append(row['AU05_r'])
        philContemptAU['AU06_r'].append(row['AU06_r'])
        philContemptAU['AU07_r'].append(row['AU07_r'])
        philContemptAU['AU09_r'].append(row['AU09_r'])
        philContemptAU['AU10_r'].append(row['AU10_r'])
        philContemptAU['AU12_r'].append(row['AU12_r'])
        philContemptAU['AU14_r'].append(row['AU14_r'])
        philContemptAU['AU15_r'].append(row['AU15_r'])
        philContemptAU['AU17_r'].append(row['AU17_r'])
        philContemptAU['AU23_r'].append(row['AU23_r'])
        philContemptAU['AU25_r'].append(row['AU25_r'])
        philContemptAU['AU26_r'].append(row['AU26_r'])
        philContemptAU['AU45_r'].append(row['AU45_r'])
    elif row['culture'] == 'Philippines' and row['emotion'] == 'anger':
        philAngerAU['AU01_r'].append(row['AU01_r'])
        philAngerAU['AU02_r'].append(row['AU02_r'])
        philAngerAU['AU04_r'].append(row['AU04_r'])
        philAngerAU['AU05_r'].append(row['AU05_r'])
        philAngerAU['AU06_r'].append(row['AU06_r'])
        philAngerAU['AU07_r'].append(row['AU07_r'])
        philAngerAU['AU09_r'].append(row['AU09_r'])
        philAngerAU['AU10_r'].append(row['AU10_r'])
        philAngerAU['AU12_r'].append(row['AU12_r'])
        philAngerAU['AU14_r'].append(row['AU14_r'])
        philAngerAU['AU15_r'].append(row['AU15_r'])
        philAngerAU['AU17_r'].append(row['AU17_r'])
        philAngerAU['AU23_r'].append(row['AU23_r'])
        philAngerAU['AU25_r'].append(row['AU25_r'])
        philAngerAU['AU26_r'].append(row['AU26_r'])
        philAngerAU['AU45_r'].append(row['AU45_r'])
    elif row['culture'] == 'Philippines' and row['emotion'] == 'disgust':
        philDisgustAU['AU01_r'].append(row['AU01_r'])
        philDisgustAU['AU02_r'].append(row['AU02_r'])
        philDisgustAU['AU04_r'].append(row['AU04_r'])
        philDisgustAU['AU05_r'].append(row['AU05_r'])
        philDisgustAU['AU06_r'].append(row['AU06_r'])
        philDisgustAU['AU07_r'].append(row['AU07_r'])
        philDisgustAU['AU09_r'].append(row['AU09_r'])
        philDisgustAU['AU10_r'].append(row['AU10_r'])
        philDisgustAU['AU12_r'].append(row['AU12_r'])
        philDisgustAU['AU14_r'].append(row['AU14_r'])
        philDisgustAU['AU15_r'].append(row['AU15_r'])
        philDisgustAU['AU17_r'].append(row['AU17_r'])
        philDisgustAU['AU23_r'].append(row['AU23_r'])
        philDisgustAU['AU25_r'].append(row['AU25_r'])
        philDisgustAU['AU26_r'].append(row['AU26_r'])
        philDisgustAU['AU45_r'].append(row['AU45_r'])


#for key in persianContemptAU:
#    print(key + ": " + str(st.stdev(persianContemptAU[key])))


meanContemptData = {'Persian': {'AU01_r': st.mean(persianContemptAU['AU01_r']), 'AU02_r': st.mean(persianContemptAU['AU02_r']),
                       'AU04_r': st.mean(persianContemptAU['AU04_r']), 'AU05_r': st.mean(persianContemptAU['AU05_r']),
                       'AU06_r': st.mean(persianContemptAU['AU06_r']), 'AU07_r': st.mean(persianContemptAU['AU07_r']),
                       'AU09_r': st.mean(persianContemptAU['AU09_r']), 'AU10_r': st.mean(persianContemptAU['AU10_r']),
                       'AU12_r': st.mean(persianContemptAU['AU12_r']), 'AU14_r': st.mean(persianContemptAU['AU14_r']),
                       'AU15_r': st.mean(persianContemptAU['AU15_r']), 'AU17_r': st.mean(persianContemptAU['AU17_r']),
                       'AU23_r': st.mean(persianContemptAU['AU23_r']), 'AU25_r': st.mean(persianContemptAU['AU25_r']),
                       'AU26_r': st.mean(persianContemptAU['AU26_r']), 'AU45_r': st.mean(persianContemptAU['AU45_r'])},
            
           'North America': {'AU01_r': st.mean(naContemptAU['AU01_r']), 'AU02_r': st.mean(naContemptAU['AU02_r']),
                       'AU04_r': st.mean(naContemptAU['AU04_r']), 'AU05_r': st.mean(naContemptAU['AU05_r']),
                       'AU06_r': st.mean(naContemptAU['AU06_r']), 'AU07_r': st.mean(naContemptAU['AU07_r']),
                       'AU09_r': st.mean(naContemptAU['AU09_r']), 'AU10_r': st.mean(naContemptAU['AU10_r']),
                       'AU12_r': st.mean(naContemptAU['AU12_r']), 'AU14_r': st.mean(naContemptAU['AU14_r']),
                       'AU15_r': st.mean(naContemptAU['AU15_r']), 'AU17_r': st.mean(naContemptAU['AU17_r']),
                       'AU23_r': st.mean(naContemptAU['AU23_r']), 'AU25_r': st.mean(naContemptAU['AU25_r']),
                       'AU26_r': st.mean(naContemptAU['AU26_r']), 'AU45_r': st.mean(naContemptAU['AU45_r'])},
            
           'Philippines': {'AU01_r': st.mean(philContemptAU['AU01_r']), 'AU02_r': st.mean(philContemptAU['AU02_r']),
                       'AU04_r': st.mean(philContemptAU['AU04_r']), 'AU05_r': st.mean(philContemptAU['AU05_r']),
                       'AU06_r': st.mean(philContemptAU['AU06_r']), 'AU07_r': st.mean(philContemptAU['AU07_r']),
                       'AU09_r': st.mean(philContemptAU['AU09_r']), 'AU10_r': st.mean(philContemptAU['AU10_r']),
                       'AU12_r': st.mean(philContemptAU['AU12_r']), 'AU14_r': st.mean(philContemptAU['AU14_r']),
                       'AU15_r': st.mean(philContemptAU['AU15_r']), 'AU17_r': st.mean(philContemptAU['AU17_r']),
                       'AU23_r': st.mean(philContemptAU['AU23_r']), 'AU25_r': st.mean(philContemptAU['AU25_r']),
                       'AU26_r': st.mean(philContemptAU['AU26_r']), 'AU45_r': st.mean(philContemptAU['AU45_r'])}}

'''
meanContemptData = {'Persian': {st.mean(persianContemptAU['AU01_r']): 'AU01_r', st.mean(persianContemptAU['AU02_r']): 'AU02_r',
                       st.mean(persianContemptAU['AU04_r']): 'AU04_r', st.mean(persianContemptAU['AU05_r']): 'AU05_r',
                       st.mean(persianContemptAU['AU06_r']): 'AU06_r', st.mean(persianContemptAU['AU07_r']): 'AU07_r',
                       st.mean(persianContemptAU['AU09_r']): 'AU09_r', st.mean(persianContemptAU['AU10_r']): 'AU10_r',
                       st.mean(persianContemptAU['AU12_r']): 'AU12_r', st.mean(persianContemptAU['AU14_r']): 'AU14_r',
                       st.mean(persianContemptAU['AU15_r']): 'AU15_r', st.mean(persianContemptAU['AU17_r']): 'AU17_r',
                       st.mean(persianContemptAU['AU23_r']): 'AU23_r', st.mean(persianContemptAU['AU25_r']): 'AU25_r',
                       st.mean(persianContemptAU['AU26_r']): 'AU26_r', st.mean(persianContemptAU['AU45_r']): 'AU45_r'},
            
           'North America': {st.mean(naContemptAU['AU01_r']): 'AU01_r', st.mean(naContemptAU['AU02_r']): 'AU02_r',
                       st.mean(naContemptAU['AU04_r']): 'AU04_r', st.mean(naContemptAU['AU05_r']): 'AU05_r',
                       st.mean(naContemptAU['AU06_r']): 'AU06_r', st.mean(naContemptAU['AU07_r']): 'AU07_r',
                       st.mean(naContemptAU['AU09_r']): 'AU09_r', st.mean(naContemptAU['AU10_r']): 'AU10_r',
                       st.mean(naContemptAU['AU12_r']): 'AU12_r', st.mean(naContemptAU['AU14_r']): 'AU14_r',
                       st.mean(naContemptAU['AU15_r']): 'AU15_r', st.mean(naContemptAU['AU17_r']): 'AU17_r',
                       st.mean(naContemptAU['AU23_r']): 'AU23_r', st.mean(naContemptAU['AU25_r']): 'AU25_r',
                       st.mean(naContemptAU['AU26_r']): 'AU26_r', st.mean(naContemptAU['AU45_r']): 'AU45_r'},
            
           'Philippines': {st.mean(philContemptAU['AU01_r']): 'AU01_r', st.mean(philContemptAU['AU02_r']): 'AU02_r',
                       st.mean(philContemptAU['AU04_r']): 'AU04_r', st.mean(philContemptAU['AU05_r']): 'AU05_r',
                       st.mean(philContemptAU['AU06_r']): 'AU06_r', st.mean(philContemptAU['AU07_r']): 'AU07_r',
                       st.mean(philContemptAU['AU09_r']): 'AU09_r', st.mean(philContemptAU['AU10_r']): 'AU10_r',
                       st.mean(philContemptAU['AU12_r']): 'AU12_r', st.mean(philContemptAU['AU14_r']): 'AU14_r',
                       st.mean(philContemptAU['AU15_r']): 'AU15_r', st.mean(philContemptAU['AU17_r']): 'AU17_r',
                       st.mean(philContemptAU['AU23_r']): 'AU23_r', st.mean(philContemptAU['AU25_r']): 'AU25_r',
                       st.mean(philContemptAU['AU26_r']): 'AU26_r', st.mean(philContemptAU['AU45_r']): 'AU45_r'}}
'''
print("Mean values for each AU for each culture with emotions labelled as contempt\n")
df_mean_contempt = pd.DataFrame(meanContemptData)

print(df_mean_contempt)

ax = df_mean_contempt.plot(kind='bar', figsize =(13,13), rot = 0)
ax.set_xlabel("AU", fontsize = 20)
ax.set_ylabel("Mean Value", fontsize = 20)
plt.title('Contempt: Mean Value of Each AU from Each Culture ', fontsize = 22)
plt.show()



meanAngerData = {'Persian': {'AU01_r': st.mean(persianAngerAU['AU01_r']), 'AU02_r': st.mean(persianAngerAU['AU02_r']),
                       'AU04_r': st.mean(persianAngerAU['AU04_r']), 'AU05_r': st.mean(persianAngerAU['AU05_r']),
                       'AU06_r': st.mean(persianAngerAU['AU06_r']), 'AU07_r': st.mean(persianAngerAU['AU07_r']),
                       'AU09_r': st.mean(persianAngerAU['AU09_r']), 'AU10_r': st.mean(persianAngerAU['AU10_r']),
                       'AU12_r': st.mean(persianAngerAU['AU12_r']), 'AU14_r': st.mean(persianAngerAU['AU14_r']),
                       'AU15_r': st.mean(persianAngerAU['AU15_r']), 'AU17_r': st.mean(persianAngerAU['AU17_r']),
                       'AU23_r': st.mean(persianAngerAU['AU23_r']), 'AU25_r': st.mean(persianAngerAU['AU25_r']),
                       'AU26_r': st.mean(persianAngerAU['AU26_r']), 'AU45_r': st.mean(persianAngerAU['AU45_r'])},
            
           'North America': {'AU01_r': st.mean(naContemptAU['AU01_r']), 'AU02_r': st.mean(naAngerAU['AU02_r']),
                       'AU04_r': st.mean(naAngerAU['AU04_r']), 'AU05_r': st.mean(naAngerAU['AU05_r']),
                       'AU06_r': st.mean(naAngerAU['AU06_r']), 'AU07_r': st.mean(naAngerAU['AU07_r']),
                       'AU09_r': st.mean(naAngerAU['AU09_r']), 'AU10_r': st.mean(naAngerAU['AU10_r']),
                       'AU12_r': st.mean(naAngerAU['AU12_r']), 'AU14_r': st.mean(naAngerAU['AU14_r']),
                       'AU15_r': st.mean(naAngerAU['AU15_r']), 'AU17_r': st.mean(naAngerAU['AU17_r']),
                       'AU23_r': st.mean(naAngerAU['AU23_r']), 'AU25_r': st.mean(naAngerAU['AU25_r']),
                       'AU26_r': st.mean(naAngerAU['AU26_r']), 'AU45_r': st.mean(naAngerAU['AU45_r'])},
            
           'Philippines': {'AU01_r': st.mean(philContemptAU['AU01_r']), 'AU02_r': st.mean(philAngerAU['AU02_r']),
                       'AU04_r': st.mean(philAngerAU['AU04_r']), 'AU05_r': st.mean(philAngerAU['AU05_r']),
                       'AU06_r': st.mean(philAngerAU['AU06_r']), 'AU07_r': st.mean(philAngerAU['AU07_r']),
                       'AU09_r': st.mean(philAngerAU['AU09_r']), 'AU10_r': st.mean(philAngerAU['AU10_r']),
                       'AU12_r': st.mean(philAngerAU['AU12_r']), 'AU14_r': st.mean(philAngerAU['AU14_r']),
                       'AU15_r': st.mean(philAngerAU['AU15_r']), 'AU17_r': st.mean(philAngerAU['AU17_r']),
                       'AU23_r': st.mean(philAngerAU['AU23_r']), 'AU25_r': st.mean(philAngerAU['AU25_r']),
                       'AU26_r': st.mean(philAngerAU['AU26_r']), 'AU45_r': st.mean(philAngerAU['AU45_r'])}}
print("\n")
print("Mean values for each AU for each culture with emotions labelled as anger\n")
df_mean_anger = pd.DataFrame(meanAngerData)

print(df_mean_anger)

meanDisgustData = {'Persian': {'AU01_r': st.mean(persianDisgustAU['AU01_r']), 'AU02_r': st.mean(persianDisgustAU['AU02_r']),
                       'AU04_r': st.mean(persianDisgustAU['AU04_r']), 'AU05_r': st.mean(persianDisgustAU['AU05_r']),
                       'AU06_r': st.mean(persianDisgustAU['AU06_r']), 'AU07_r': st.mean(persianDisgustAU['AU07_r']),
                       'AU09_r': st.mean(persianDisgustAU['AU09_r']), 'AU10_r': st.mean(persianDisgustAU['AU10_r']),
                       'AU12_r': st.mean(persianDisgustAU['AU12_r']), 'AU14_r': st.mean(persianDisgustAU['AU14_r']),
                       'AU15_r': st.mean(persianDisgustAU['AU15_r']), 'AU17_r': st.mean(persianDisgustAU['AU17_r']),
                       'AU23_r': st.mean(persianDisgustAU['AU23_r']), 'AU25_r': st.mean(persianDisgustAU['AU25_r']),
                       'AU26_r': st.mean(persianDisgustAU['AU26_r']), 'AU45_r': st.mean(persianDisgustAU['AU45_r'])},
            
           'North America': {'AU01_r': st.mean(naDisgustAU['AU01_r']), 'AU02_r': st.mean(naDisgustAU['AU02_r']),
                       'AU04_r': st.mean(naDisgustAU['AU04_r']), 'AU05_r': st.mean(naDisgustAU['AU05_r']),
                       'AU06_r': st.mean(naDisgustAU['AU06_r']), 'AU07_r': st.mean(naDisgustAU['AU07_r']),
                       'AU09_r': st.mean(naDisgustAU['AU09_r']), 'AU10_r': st.mean(naDisgustAU['AU10_r']),
                       'AU12_r': st.mean(naDisgustAU['AU12_r']), 'AU14_r': st.mean(naDisgustAU['AU14_r']),
                       'AU15_r': st.mean(naDisgustAU['AU15_r']), 'AU17_r': st.mean(naDisgustAU['AU17_r']),
                       'AU23_r': st.mean(naDisgustAU['AU23_r']), 'AU25_r': st.mean(naDisgustAU['AU25_r']),
                       'AU26_r': st.mean(naDisgustAU['AU26_r']), 'AU45_r': st.mean(naDisgustAU['AU45_r'])},
            
           'Philippines': {'AU01_r': st.mean(philDisgustAU['AU01_r']), 'AU02_r': st.mean(philDisgustAU['AU02_r']),
                       'AU04_r': st.mean(philDisgustAU['AU04_r']), 'AU05_r': st.mean(philDisgustAU['AU05_r']),
                       'AU06_r': st.mean(philDisgustAU['AU06_r']), 'AU07_r': st.mean(philDisgustAU['AU07_r']),
                       'AU09_r': st.mean(philDisgustAU['AU09_r']), 'AU10_r': st.mean(philDisgustAU['AU10_r']),
                       'AU12_r': st.mean(philDisgustAU['AU12_r']), 'AU14_r': st.mean(philDisgustAU['AU14_r']),
                       'AU15_r': st.mean(philDisgustAU['AU15_r']), 'AU17_r': st.mean(philDisgustAU['AU17_r']),
                       'AU23_r': st.mean(philDisgustAU['AU23_r']), 'AU25_r': st.mean(philDisgustAU['AU25_r']),
                       'AU26_r': st.mean(philDisgustAU['AU26_r']), 'AU45_r': st.mean(philDisgustAU['AU45_r'])}}

print("\n")
print("Mean values for each AU for each culture with emotions labelled as disgust\n")
df_mean_disgust = pd.DataFrame(meanDisgustData)

print(df_mean_disgust)

stdevContemptData = {'Persian': {'AU01_r': st.stdev(persianContemptAU['AU01_r']), 'AU02_r': st.stdev(persianContemptAU['AU02_r']),
                       'AU04_r': st.stdev(persianContemptAU['AU04_r']), 'AU05_r': st.stdev(persianContemptAU['AU05_r']),
                       'AU06_r': st.stdev(persianContemptAU['AU06_r']), 'AU07_r': st.stdev(persianContemptAU['AU07_r']),
                       'AU09_r': st.stdev(persianContemptAU['AU09_r']), 'AU10_r': st.stdev(persianContemptAU['AU10_r']),
                       'AU12_r': st.stdev(persianContemptAU['AU12_r']), 'AU14_r': st.stdev(persianContemptAU['AU14_r']),
                       'AU15_r': st.stdev(persianContemptAU['AU15_r']), 'AU17_r': st.stdev(persianContemptAU['AU17_r']),
                       'AU23_r': st.stdev(persianContemptAU['AU23_r']), 'AU25_r': st.stdev(persianContemptAU['AU25_r']),
                       'AU26_r': st.stdev(persianContemptAU['AU26_r']), 'AU45_r': st.stdev(persianContemptAU['AU45_r'])},
            
           'North America': {'AU01_r': st.stdev(naContemptAU['AU01_r']), 'AU02_r': st.stdev(naContemptAU['AU02_r']),
                       'AU04_r': st.stdev(naContemptAU['AU04_r']), 'AU05_r': st.stdev(naContemptAU['AU05_r']),
                       'AU06_r': st.stdev(naContemptAU['AU06_r']), 'AU07_r': st.stdev(naContemptAU['AU07_r']),
                       'AU09_r': st.stdev(naContemptAU['AU09_r']), 'AU10_r': st.stdev(naContemptAU['AU10_r']),
                       'AU12_r': st.stdev(naContemptAU['AU12_r']), 'AU14_r': st.stdev(naContemptAU['AU14_r']),
                       'AU15_r': st.stdev(naContemptAU['AU15_r']), 'AU17_r': st.stdev(naContemptAU['AU17_r']),
                       'AU23_r': st.stdev(naContemptAU['AU23_r']), 'AU25_r': st.stdev(naContemptAU['AU25_r']),
                       'AU26_r': st.stdev(naContemptAU['AU26_r']), 'AU45_r': st.stdev(naContemptAU['AU45_r'])},
            
           'Philippines': {'AU01_r': st.stdev(philContemptAU['AU01_r']), 'AU02_r': st.stdev(philContemptAU['AU02_r']),
                       'AU04_r': st.stdev(philContemptAU['AU04_r']), 'AU05_r': st.stdev(philContemptAU['AU05_r']),
                       'AU06_r': st.stdev(philContemptAU['AU06_r']), 'AU07_r': st.stdev(philContemptAU['AU07_r']),
                       'AU09_r': st.stdev(philContemptAU['AU09_r']), 'AU10_r': st.stdev(philContemptAU['AU10_r']),
                       'AU12_r': st.stdev(philContemptAU['AU12_r']), 'AU14_r': st.stdev(philContemptAU['AU14_r']),
                       'AU15_r': st.stdev(philContemptAU['AU15_r']), 'AU17_r': st.stdev(philContemptAU['AU17_r']),
                       'AU23_r': st.stdev(philContemptAU['AU23_r']), 'AU25_r': st.stdev(philContemptAU['AU25_r']),
                       'AU26_r': st.stdev(philContemptAU['AU26_r']), 'AU45_r': st.stdev(philContemptAU['AU45_r'])}}

print("Standard deviation values for each AU for each culture with emotions labelled as contempt\n")
df_stdev_contempt = pd.DataFrame(stdevContemptData)

print(df_stdev_contempt)

ax = df_stdev_contempt.plot(kind='bar', figsize =(13,13), rot = 0)
ax.set_xlabel("AU", fontsize = 20)
ax.set_ylabel("Standard Deviation Value", fontsize = 20)
plt.title('Contempt: Standard Deviation Value of Each AU from Each Culture ', fontsize = 22)
plt.show()

stdevAngerData = {'Persian': {'AU01_r': st.stdev(persianAngerAU['AU01_r']), 'AU02_r': st.stdev(persianAngerAU['AU02_r']),
                       'AU04_r': st.stdev(persianAngerAU['AU04_r']), 'AU05_r': st.stdev(persianAngerAU['AU05_r']),
                       'AU06_r': st.stdev(persianAngerAU['AU06_r']), 'AU07_r': st.stdev(persianAngerAU['AU07_r']),
                       'AU09_r': st.stdev(persianAngerAU['AU09_r']), 'AU10_r': st.stdev(persianAngerAU['AU10_r']),
                       'AU12_r': st.stdev(persianAngerAU['AU12_r']), 'AU14_r': st.stdev(persianAngerAU['AU14_r']),
                       'AU15_r': st.stdev(persianAngerAU['AU15_r']), 'AU17_r': st.stdev(persianAngerAU['AU17_r']),
                       'AU23_r': st.stdev(persianAngerAU['AU23_r']), 'AU25_r': st.stdev(persianAngerAU['AU25_r']),
                       'AU26_r': st.stdev(persianAngerAU['AU26_r']), 'AU45_r': st.stdev(persianAngerAU['AU45_r'])},
            
           'North America': {'AU01_r': st.stdev(naContemptAU['AU01_r']), 'AU02_r': st.stdev(naAngerAU['AU02_r']),
                       'AU04_r': st.stdev(naAngerAU['AU04_r']), 'AU05_r': st.stdev(naAngerAU['AU05_r']),
                       'AU06_r': st.stdev(naAngerAU['AU06_r']), 'AU07_r': st.stdev(naAngerAU['AU07_r']),
                       'AU09_r': st.stdev(naAngerAU['AU09_r']), 'AU10_r': st.stdev(naAngerAU['AU10_r']),
                       'AU12_r': st.stdev(naAngerAU['AU12_r']), 'AU14_r': st.stdev(naAngerAU['AU14_r']),
                       'AU15_r': st.stdev(naAngerAU['AU15_r']), 'AU17_r': st.stdev(naAngerAU['AU17_r']),
                       'AU23_r': st.stdev(naAngerAU['AU23_r']), 'AU25_r': st.stdev(naAngerAU['AU25_r']),
                       'AU26_r': st.stdev(naAngerAU['AU26_r']), 'AU45_r': st.stdev(naAngerAU['AU45_r'])},
            
           'Philippines': {'AU01_r': st.stdev(philContemptAU['AU01_r']), 'AU02_r': st.stdev(philAngerAU['AU02_r']),
                       'AU04_r': st.stdev(philAngerAU['AU04_r']), 'AU05_r': st.stdev(philAngerAU['AU05_r']),
                       'AU06_r': st.stdev(philAngerAU['AU06_r']), 'AU07_r': st.stdev(philAngerAU['AU07_r']),
                       'AU09_r': st.stdev(philAngerAU['AU09_r']), 'AU10_r': st.stdev(philAngerAU['AU10_r']),
                       'AU12_r': st.stdev(philAngerAU['AU12_r']), 'AU14_r': st.stdev(philAngerAU['AU14_r']),
                       'AU15_r': st.stdev(philAngerAU['AU15_r']), 'AU17_r': st.stdev(philAngerAU['AU17_r']),
                       'AU23_r': st.stdev(philAngerAU['AU23_r']), 'AU25_r': st.stdev(philAngerAU['AU25_r']),
                       'AU26_r': st.stdev(philAngerAU['AU26_r']), 'AU45_r': st.stdev(philAngerAU['AU45_r'])}}

print("Standard deviation values for each AU for each culture with emotions labelled as anger\n")
df_stdev_anger = pd.DataFrame(stdevAngerData)

print(df_stdev_anger)

stdevDisgustData = {'Persian': {'AU01_r': st.stdev(persianDisgustAU['AU01_r']), 'AU02_r': st.stdev(persianDisgustAU['AU02_r']),
                       'AU04_r': st.stdev(persianDisgustAU['AU04_r']), 'AU05_r': st.stdev(persianDisgustAU['AU05_r']),
                       'AU06_r': st.stdev(persianDisgustAU['AU06_r']), 'AU07_r': st.stdev(persianDisgustAU['AU07_r']),
                       'AU09_r': st.stdev(persianDisgustAU['AU09_r']), 'AU10_r': st.stdev(persianDisgustAU['AU10_r']),
                       'AU12_r': st.stdev(persianDisgustAU['AU12_r']), 'AU14_r': st.stdev(persianDisgustAU['AU14_r']),
                       'AU15_r': st.stdev(persianDisgustAU['AU15_r']), 'AU17_r': st.stdev(persianDisgustAU['AU17_r']),
                       'AU23_r': st.stdev(persianDisgustAU['AU23_r']), 'AU25_r': st.stdev(persianDisgustAU['AU25_r']),
                       'AU26_r': st.stdev(persianDisgustAU['AU26_r']), 'AU45_r': st.stdev(persianDisgustAU['AU45_r'])},
            
           'North America': {'AU01_r': st.stdev(naDisgustAU['AU01_r']), 'AU02_r': st.stdev(naDisgustAU['AU02_r']),
                       'AU04_r': st.stdev(naDisgustAU['AU04_r']), 'AU05_r': st.stdev(naDisgustAU['AU05_r']),
                       'AU06_r': st.stdev(naDisgustAU['AU06_r']), 'AU07_r': st.stdev(naDisgustAU['AU07_r']),
                       'AU09_r': st.stdev(naDisgustAU['AU09_r']), 'AU10_r': st.stdev(naDisgustAU['AU10_r']),
                       'AU12_r': st.stdev(naDisgustAU['AU12_r']), 'AU14_r': st.stdev(naDisgustAU['AU14_r']),
                       'AU15_r': st.stdev(naDisgustAU['AU15_r']), 'AU17_r': st.stdev(naDisgustAU['AU17_r']),
                       'AU23_r': st.stdev(naDisgustAU['AU23_r']), 'AU25_r': st.stdev(naDisgustAU['AU25_r']),
                       'AU26_r': st.stdev(naDisgustAU['AU26_r']), 'AU45_r': st.stdev(naDisgustAU['AU45_r'])},
            
           'Philippines': {'AU01_r': st.stdev(philDisgustAU['AU01_r']), 'AU02_r': st.stdev(philDisgustAU['AU02_r']),
                       'AU04_r': st.stdev(philDisgustAU['AU04_r']), 'AU05_r': st.stdev(philDisgustAU['AU05_r']),
                       'AU06_r': st.stdev(philDisgustAU['AU06_r']), 'AU07_r': st.stdev(philDisgustAU['AU07_r']),
                       'AU09_r': st.stdev(philDisgustAU['AU09_r']), 'AU10_r': st.stdev(philDisgustAU['AU10_r']),
                       'AU12_r': st.stdev(philDisgustAU['AU12_r']), 'AU14_r': st.stdev(philDisgustAU['AU14_r']),
                       'AU15_r': st.stdev(philDisgustAU['AU15_r']), 'AU17_r': st.stdev(philDisgustAU['AU17_r']),
                       'AU23_r': st.stdev(philDisgustAU['AU23_r']), 'AU25_r': st.stdev(philDisgustAU['AU25_r']),
                       'AU26_r': st.stdev(philDisgustAU['AU26_r']), 'AU45_r': st.stdev(philDisgustAU['AU45_r'])}}

print("Standard deviation values for each AU for each culture with emotions labelled as disgust\n")
df_stdev_disgust = pd.DataFrame(stdevDisgustData)

print(df_stdev_disgust)

        