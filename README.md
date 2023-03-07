# Lung Cancer Detection



Lung cancer is a leading cause of cancer deaths worldwide, and early detection is crucial for improving survival rates. In this project, we aimed to develop a feature extraction model for predicting lung cancer and to compare it with other state-of-the-art models.

The first step in our process was to review various existing lung cancer prediction models and identify their strengths and weaknesses. We then developed a new feature extraction model that combines the strengths of multiple existing models.

Next, we compared the symptoms of lung cancer to identify patterns that could be used for early detection. This included analyzing medical records and conducting surveys with patients to gather data on common symptoms such as cough, shortness of breath, and chest pain.

We also designed and developed a deep learning model for predicting lung cancer. This model was trained on a large dataset of medical images and patient information, and was able to accurately predict the presence of lung cancer in a majority of cases.

Basically, This project aims to develop a feature extraction model for predicting lung cancer and to compare it with other state-of-the-art models. We also designed and developed a deep learning model for predicting lung cancer and implemented an automatic notification system for early detection.

## Demo

https://user-images.githubusercontent.com/63944541/223366523-27406f06-2c3f-4778-8900-088afb26f780.mp4


## Dataset:

Images are not in dcm format, the images are in jpg or png to fit the model
Data contain 3 chest cancer types which are Adenocarcinoma,Large cell carcinoma, Squamous cell carcinoma , and 1 folder for the normal cell
Data folder is the main folder that contain all the step folders
inside Data folder are test , train , valid
Downloaded from: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

- test represent testing set
- train represent training set
- valid represent validation set
- training set is 70%
- testing set is 20%
- validation set is 10%

1. **Adenocarcinoma**
Adenocarcinoma of the lung: Lung adenocarcinoma is the most common form of lung cancer accounting for 30 percent of all cases overall and about 40 percent
of all non-small cell lung cancer occurrences. Adenocarcinomas are found in several common cancers, including breast, prostate and colorectal.
Adenocarcinomas of the lung are found in the outer region of the lung in glands that secrete mucus and help us breathe.
Symptoms include coughing, hoarseness, weight loss and weakness.

2. **Large cell carcinoma**
Large-cell undifferentiated carcinoma: Large-cell undifferentiated carcinoma lung cancer grows and spreads quickly and can be found anywhere in the lung. This type of lung cancer usually accounts for 10 to 15 percent of all cases of NSCLC.
Large-cell undifferentiated carcinoma tends to grow and spread quickly.

3. **Squamous cell carcinoma**
Squamous cell: This type of lung cancer is found centrally in the lung, where the larger bronchi join the trachea to the lung, or in one of the main airway branches.
Squamous cell lung cancer is responsible for about 30 percent of all non-small cell lung cancers, and is generally linked to smoking.

4. **Normal**
And the last folder is the normal CT-Scan images

### Process:

The process of our project can be divided into three main steps:

1. **Processing:** In this step, we collected and processed data from medical records and surveys with patients to identify patterns in the symptoms of lung cancer.

2. **Feature extraction:** In this step, we developed a new feature extraction model by combining the strengths of multiple existing models.

3. **Prediction:** In this step, we used the feature extraction model to predict the presence of lung cancer in patients.

### Classification:

We used several classification algorithms to compare our hybrid approach with other conventional models, including:

- K-Nearest Neighbors Algorithm
- Support Vector Machines (SVMs)
- Decision Tree
- Artificial Neural Networks (ANNs)
- Convolutional Neural Networks (CNNs)
- Hybrid approach 1 (SVM+KNN)
- Hybrid approach 2 (SVM+ ANN)

### Performance Parameters

To evaluate the performance of our models, we used a range of performance parameters including:

1. Accuracy
2. Sensitivity
3. Specificity
4. Recall
5. F-Measure
6. Error
7. Precision
8. False Positive Rate


## Automatic Notification
We implemented an automatic notification system that could alert medical professionals if a patient was at high risk for lung cancer. This system was designed to be used in conjunction with our prediction models and was able to significantly improve the speed and accuracy of diagnosis.


## Features

- Drag and drop images 
- It predicts the input image for the classes
- Predicts the accuracy with 7 different classifiers for the input image
- Each Classifier provides the metrics for particular prediction with graph



## Tech

The website uses a number of open source projects to work properly:

- [Tensorflow] - Deep learning application framework
- [Scikit-Learn] - Bank for classification, predictive analytics, and very many other machine learning tasks.
- [Flask] - Framework for creating web applications in Python easier.
- [Matplotlib] - A low level graph plotting library in python that serves as a visualization utility.
- [Numpy] - Used for working with arrays
- [Pandas] - Used for data analysis and associated manipulation of tabular data in Dataframes

## Screenshots and Steps

**1. Landing Page:**

- This is the landing page for the web application 

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/1.png?raw=true)

**2. Upload button:**
 
- Later on the web application it provides 3 different buttons along with a upload button where you upload your input image and later it provides you with 3 buttons of Preprocessing, Feature Extraction and Prediction of the uploaded image

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/2.png?raw=true)

**3. Preprocessing:**


- After uploading the image the image needs to be preprocessed where it is preprocessing using two techniques which is Resizing of the uploaded input image from it's original size to the size which is required for the image to predict on.
- After resizing of the image Data Augmentation is applied on the input image 
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/3.png?raw=true)

 


**5. Feature Extraction :**

- After Preprocessing comes the part of Feature Extraction where we extract important features of the input image by converting the uploaded image from Original to Grayscale and pointing out the important parts required for the model to predict the following image

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/4.png?raw=true)

**6. Classifiers :**

- After the Feature Extraction comes the part of Prediction where the project is trained on 7 different classifiers which are SVM, KNN, ANN, DT, CNN, Hybrid(SVM+ANN) and it displays it's prediction on those 7 classifiers along with the confidence at which it has predicted the following image 
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/5.png?raw=true)

- If you click on any of the classifiers it further shows you the classification metrics on that particular classifier along with the visualization of that model
-![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/6.png?raw=true)

**7. Comparitive Analysis :**
- At last the application provides you with a comparitive analysis of all the classifiers where you can compare the accuracy of each classfier side by side in the format of table as well as graph
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/086dc5ebb47bfd6e1ee440d254edc1242e8312ee/Lung%20Cancer/7.png?raw=true)

## Results
Our hybrid approach consistently outperformed the other models in terms of accuracy and other performance metrics. The automatic notification system we developed has the potential to greatly improve the chances of early detection and successful treatment of lung cancer.




## Installation

Website requires these steps to install the application on your device


On terminal:

Download virtual env library
```sh
pip3 install -U pip virtualenv
```

Create a virtual environment on your device
```sh
virtualenv  -p python3 ./venv
```

Download all the dependencies provided on requirements.txt
```sh
pip install -r .\requirements.txt
```

Activated the virtual environment
```sh
.\pp\Scripts\activate
```

Run app.py after completing all the steps.





[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   
[Tensorflow]: <https://www.tensorflow.org/>
[Scikit-Learn]: <https://scikit-learn.org/stable/>
[Flask]: <https://flask.palletsprojects.com/en/2.1.x/>
[Matplotlib]: <https://matplotlib.org/>
[Numpy]: <https://numpy.org/>
[Pandas]: <https://pandas.pydata.org/>


