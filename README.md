# EX-01 Introduction to Kaggle and Data preprocessing
### DATE:25.02.2024
### AIM:
To perform Data preprocessing in a data set downloaded from Kaggle. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
### EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
### RELATED THEORETICAL CONCEPT:
**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.<br><br>
**Data Preprocessing:**
Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing.<br><br>
**Need of Data Preprocessing :**
For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.

### ALGORITHM:
STEP 1:Importing the libraries.<BR>
STEP 2:Importing the dataset.<BR>
STEP 3:Taking care of missing data<br>
STEP 4:Encoding categorical data.<br>
STEP 5:Normalizing the data.<BR>
STEP 6:Splitting the data into test and train.<BR>
**Developed By: PAVANA.G**<br>
**Register No: 212222230105**<br>
###  PROGRAM:
```Python
import pandas as pd                                                 # Importing Libraries
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         # Read the dataset from drive
df.head()
df.isnull().sum()                                                   # Finding Missing Values
df.duplicated().sum()                                               # Check For Duplicates
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               # Remove Unnecessary Columns
scaler=StandardScaler()                                             # Normalize the dataset
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     # Split the dataset into input and output
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   # Splitting the data for training & Testing
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                     # X Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   # Y Train and Test
```
### OUTPUT:
**DATASET:** <br>
<img eight=45% idth=34% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/f7360f2e-bc07-40f9-aee2-9c17a02bcb25"><br>\
<br>
**NULL VALUES:** &emsp;&emsp;&emsp;&emsp;&emsp; **NORMALIZED DATA:** <br>
<img eight=45% idth=34% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/f8da2d3d-b0a5-4f5d-9b1e-dce058b9f4b4">
<img eight=45% idth=34% align=top src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/557ce4de-82fe-4b3b-9b17-83029f74d37f"><br>
**DATA SPLITTING:** <br>
<img eight=45% width=49% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/d312c539-39be-4335-9fda-375f3c648a5d">
<img eight=45% width=50% align=top src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/fa9f2f38-12f5-4151-8aed-5c5e128797e1"><br>
**TRAIN AND TEST DATA:** <br>
<img eight=45% width=49% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/55ba52a1-02e3-4565-afdc-33cc67d0bc27">
<img eight=45% width=49% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/b12727ea-e8af-48dd-b190-64f4fc4afd29">
<img eight=45% width=49% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/253722ea-1e06-429d-be3a-8efcb7b29d99">
<img eight=45% width=49% src="https://github.com/ROHITJAIND/EX-01-Kaggle-and-Data-preprocessing/assets/118707073/950ef76b-d298-4c62-8912-7482d3d7d1d7">

### RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


