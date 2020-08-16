# Contempt Detection in Using Videos from Different Cultures
In this project we examined different methods to classify anger, contempt and disgust.

## Approach
We used SVM and DNN to classify each frame into 3 categories.

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
1. To run the code, please ensure all_videos.csv and the images dataset (images is provided as a seperate folder because it is 750mb big) is present in the same directory as image_classification.ipynb
2. Open the notebook image_classification.ipynb in Jupyter Notebook
3. Ensure sklearn, re, os, pandas, glob and IPython.display are installed in Jupyter Notebook.
4. Run the cell containing the code

 
### Other Files
