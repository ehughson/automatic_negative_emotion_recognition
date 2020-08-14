### This code computes the emotion count of each culture ###
### Please ensure that all_videos.csv is located in the same directory as this file ###


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("all_videos.csv")

previousFilename = None
totalVideoCount = 0

persianAngerVideoCount = 0
persianContemptVideoCount = 0
persianDisgustVideoCount = 0

philippinesAngerVideoCount = 0
philippinesContemptVideoCount = 0
philippinesDisgustVideoCount = 0

naAngerVideoCount = 0
naContemptVideoCount = 0
naDisgustVideoCount = 0



for index, row in df.iterrows():  
    if index == 0:
        previousFilename = row['filename']
        
    if previousFilename != row['filename']:
        #print("Previous filename: " + previousFilename)
        previousFilename = row['filename']
        totalVideoCount += 1
        #print("Index: " + str(index))  
        #print("Filename: " +row['filename'])
        
        if row['culture'] == 'Persian' and row['emotion'] == 'contempt':
            persianContemptVideoCount += 1
        elif row['culture'] == 'Persian' and row['emotion'] == 'anger':
            persianAngerVideoCount += 1
        elif row['culture'] == 'Persian' and row['emotion'] == 'disgust':
            persianDisgustVideoCount += 1
            
        if row['culture'] == 'North America' and row['emotion'] == 'contempt':
            naContemptVideoCount += 1
        elif row['culture'] == 'North America' and row['emotion'] == 'anger':
            naAngerVideoCount += 1
        elif row['culture'] == 'North America' and row['emotion'] == 'disgust':
            naDisgustVideoCount += 1
            
        if row['culture'] == 'Philippines' and row['emotion'] == 'contempt':
            philippinesContemptVideoCount += 1
        elif row['culture'] == 'Philippines' and row['emotion'] == 'anger':
            philippinesAngerVideoCount += 1
        elif row['culture'] == 'Philippines' and row['emotion'] == 'disgust':
            philippinesDisgustVideoCount += 1
            
        
    
    
#print(persianDisgustVideoCount)
print("Total video count: " + str(totalVideoCount))
        
data = {'Persian': {'anger': persianAngerVideoCount, 'contempt': persianContemptVideoCount, 'disgust': persianDisgustVideoCount},
       'North America': {'anger': naAngerVideoCount, 'contempt': naContemptVideoCount, 'disgust': naDisgustVideoCount},
       'Philippines': {'anger': philippinesAngerVideoCount, 'contempt': philippinesContemptVideoCount, 'disgust': philippinesDisgustVideoCount}}

df = pd.DataFrame(data)
print(df)

ax = df.plot(kind='bar', figsize =(13,13), rot = 0)
ax.set_xlabel("Emotion", fontsize = 20)
ax.set_ylabel("Emotion Count", fontsize = 20)
plt.title('Number of Each Emotion from Each Culture', fontsize = 22)
plt.show()

#plt.figure(figsize=(10,10))

