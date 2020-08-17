# Contempt Detection in Using Videos from Different Cultures
In this project we examined different methods to classify anger, contempt and disgust.

## Self-evaluation of the Project with respect to the Proposal
In general, we have done everything according to our proposal. In some parts like data acquisition, we surpassed what we said in the proposal and collected more data. We did not change anything. We wanted to do more experiments but due to the time limit we could not.
The detail is as follows: 

### Dataset
In our proposal, we planned to collect at least 200 videos, with roughly 70 videos for each culture (North American, Persian, Fillipino). We managed to collect 235 videos (75 for Persian, 87 for North American and 74 for Filipino). We have at least 22 videos for each emotion in each culture. 


### Approach
We used SVM and DNN to classify each frame into 3 categories as stated in the proposal. We fed the AUs obtained from videos using OpenFace into the models to classify each video frame.

### Experiments and Evaluation
We evaluated the models via accuracy and F1-score metrics, according to the proposal. We trained both models cross-culturaly. DNN achieved 65.5% accuracy and 0.649 F1-score on validation set. It got 48.7% accuracy and 0.451 F1-score on test data as well.

The SVM model got poor result on test data. It achieved 40.6 test accuracy and 0.288 F1 score.

The DNN model proved to have better result in comparison to SVM. We also conducted an experiment to see if the models trained only on two cultures can generalize to the third one. But the accuracy in this experiment barely surpassed 30% for both DNN and SVM.

## How to run the project
First install the requirements.txt:  
```pip install -r requirements.txt```

### Running DNN
Make sure the all_videos.csv file is in the same directory as ```train_cv.py```
```python train_cv.py```  
At the end of the execution, test and validation accuracy is printed.

### Running SVM
1. To run the code, make sure the all_videos.csv file is in the same directory as SVM.py
2. Open a terminal and navigate to the directory with SVM.py and all_videos.csv
3. ensure sklearn, pandas, and numpy are installed
4. Run the command ```python3 SVM.py```

### Running Emotion Count
1. To run the code, please ensure all_videos.csv is present in the same directory as emotion_count.ipynb
2. Open the notebook emotion_count.ipynb in Jupyter Notebook
3. Ensure pandas is installed in Jupyter notebook 
4. Run the cell containing the code

### Running meanstdev
1. To run the code, please ensure all_videos.csv is present in the same directory as meanstdev.ipynb
2. Open the notebook meanstdev.ipynb in Jupyter Notebook
3. Ensure the pandas and statistics python modules are install in Jupyter Notebook
4. Run the cell containing the code

### Running Image Classification
1. To run the code, please ensure all_videos.csv and the images dataset. Images zip file is available in https://drive.google.com/file/d/1qjyVVSK2Y-oo06_gW8dj_3VQbogF3T63/view?usp=sharing  because it is 750mb big. They should be present in the same directory as image_classification.ipynb
2. Open the notebook image_classification.ipynb in Jupyter Notebook
3. Ensure sklearn, re, os, pandas, glob and IPython.display are installed in Jupyter Notebook.
4. Run the cell containing the code

 
### Evaluation
Everything that was mentioned in the proposal was completed. All the dataset was curated manually by each team member. However, we did not rely on outside datasets which we hope we could do in the future as it would provide more validation. Other than that there were no changes or deviations from the original plan. 
