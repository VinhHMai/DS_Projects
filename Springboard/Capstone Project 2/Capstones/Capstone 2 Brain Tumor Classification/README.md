# Springboard Capstone 2: Brain Tumor Classification using MRI images

![MRI](https://cdn-prod.medicalnewstoday.com/content/images/articles/264/264771/child-having-mri-scan.jpg)
MRI brain scans are actively used in medicine to detect brain tumors in patients. It has been scientifically proven that early detection can limit further brain damage, reduce the spread of cancerous cells to other parts of the body, and other potentially fatal health issues. Being able to quickly classify the type of brain tumor can help expedite treatment and reduce long term issues. The model must be able to classify brain tumors with a 95% accuracy using the provided 3064 MRI images as a template. The model will produce a baseline which can then be used on future MRI scans to quickly categorize any possible growths. 
	This project will focus on using a sample size of 3064 MRI images to create a predictive model. Using python and machine learning , insight on brain tumors and potentially malignant cancers can reduce potential injury or fatality of patients. The database manager, Jun Cheng, will provide the database from which the project will stem. 

## Background
The three types of tumors that will be observed are Meningiomas, Gliomas and Pituitary tumors. Meningiomas are mostly benign. They are extrinsic cerebral tumors that do not infiltrate the surrounding parenchyma. They are located on the Meningeal layer of the brain that cover the brain and spinal cord, and are one of the most common forms of brain tumors (accounting for ~20%). Gliomas on the other hand, are malignant tumors, which are intrinsic cerebral tumors that may infiltrate edema. These tumors are found in the central nervous system and the peripheral nervous system. They form out of Glial cells which regulate the surrounding, insulating, feeding repair and protection of neurons throughout the nervous system. Lastly, Pituitary tumors occur in the anterior body of the pituitary gland. This gland is a small gland located behind the bridge of the nose. It produces hormones that help regulate the functions of the other endocrine glands. A tumor in this gland can cause deregulation of the production of some hormones. 

## 1. Data
Data citation:
Cheng, Jun. "brain tumor dataset." Figshare. Dataset posted on 02/04/2017. https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

This dataset contains 3064 files in matlab format (.mat). Each file has 5 pieces of information. The patient’s ID number (PID), an image of the MRI brain scan (image), the type of tumor that has been identified (label), a vector storing the coordinates of discrete points on the tumor border(tumorBorder) and a masked image of the tumor (tumorMask). The ‘PID’ is a 10 digit number. The tumor image and mask are in binary image form where 1 indicates the location of the tumor. The tumor type is labeled as 1 for meningioma, 2 for glioma, and 3 for pituitary tumor. The tumorBorder contains [x1, y1, x2, y2,...] in which x1, y1 are planar coordinates on the tumor border.

## 2. Method
The Matlab files were loaded to the notebook using H5py. All of the data was then extracted into a Pandas data table. The images were extracted and saved to .png files for ease of manipulation. The dataset had a sklearn 85:15 training:test split applied. The images and labels were put into a Tensorflow.keras ImageDataGenerator Machine learning algorithm to be preprocessed. TranseferLearning and ResNet50 is used to add layers to the model to optimize the performance. The model categorically predicts the type of tumor that exists in the images in question. Using sklearn.metrics, the accuracy, precision, recall and F1-scores are calculated for the model. 

![image](https://user-images.githubusercontent.com/86093430/187093535-5d4dae3c-4344-4b5f-b81d-95ae74abed84.png)

## 3. [Data Wrangling & Cleaning](https://github.com/VinhHMai/Springboard/blob/main/Capstones/Capstone%202%20Brain%20Tumor%20Classification/Data%20Wrangling.ipynb)
As the data collected from the Matlab files were fairly complete, there was not much Data Cleaning to be done. Most of the cleaning was done as an ease of use step. The .mat files names were changed locally to facilitate organization and faster calling. The data from the .mat files were extracted onto a Pandas DataFrame, containing 3064 rows with 6 columns (File Name, Patient ID, Labels, Tumor Type, MRI .png File Location, and Mask .png File Location). The images from the .mat file were converted to .png files. There were no unfinished files (containing NaN or empty data). This was all done for easier application to the model.

## 4. [Exploratory Data Analysis](https://github.com/VinhHMai/Springboard/blob/main/Capstones/Capstone%202%20Brain%20Tumor%20Classification/Exploratory%20Data%20Analysis.ipynb)
There were 3064 .mat files contained in the data. The explored data from the .mat files provided insight that each file contained 5 pieces of information. The patient’s ID number (PID), an image of the MRI brain scan (image), the type of tumor that has been identified (label), a vector storing the coordinates of discrete points on the tumor border(tumorBorder) and a masked image of the tumor (tumorMask). The images were in binary format. They each contained 1 MRI brain scan image and 1 corresponding Mask image of the tumor. The tumor types are labeled as 1 for meningioma, 2 for glioma, and 3 for pituitary tumor within the .mat files. All files contained a segment of a tumor and none were healthy brains (with no tumor). This led to the decision to create a classification model rather than an identification model.

## 5. [Algorithms & Machine Learning](https://github.com/VinhHMai/Springboard/blob/main/Capstones/Capstone%202%20Brain%20Tumor%20Classification/Feature%20Engineering%20and%20Modeling.ipynb).
The model used was Tensorflow.keras. The Data was split into a training/test set using sklean.model_selection’s train_test_split. An ImageDataGenerator was used to split the DataFrame into a 3rd class. This further split the training data with a 15% validation portion. The over ratio is now 72.25% Training data, 15% Test data, 12.75% Validation data. The model is now trained using Transfer Learning and ResNet50. Transfer Learning is a technique used to pre-train models. This helps reduce the development time and increase performance of the model. ResNet (Residual Network) is a ANN that can be used to train the model on an ImageNet dataset. The ResNet50 model has 48 Convolution layers, 1 MaxPool and 1 Average Pool layer. The batch size of the model was set to 32 over 100 epochs. For 3064 samples, this means there were 69 steps per epoch.

## 6. Predictions & Conclusions
[Model Metrics](https://docs.google.com/spreadsheets/d/14wihSA1uJ4QmqyO1rp2aXQzkarIwobj37dVSBWweUQo/edit?usp=sharing)
Using Transfer Learning and ResNet50 was able to predict the classification of the MRI brain scans as being Meningiomas, Gliomas or Pituitary tumors with >95% accuracy with a loss of 0.0919. The model had a val_loss and val_accuracy of 0.2189 & 0.9401, respectively. This shows that the model fits the data well, which means the model build is learning and works properly. 

## 7. Future Improvements
- On future projects, a number of improvements could be made to optimize the performance of the model. Find a way to improve the speed at which the model can fit.
- Compare different models to find the best performing one.
- Add layers to improve accuracy and validation loss.
- Find a way to improve the speed at which the model can fit.

## 8. Credits
1. [Analytics Vidhya - Provided the basis for the model.](https://www.analyticsvidhya.com/blog/2021/06/brain-tumor-detection-and-localization-using-deep-learning-part-1/)
2. Cheng, Jun. - ["brain tumor dataset."](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) Figshare. Dataset posted on 02/04/2017. 