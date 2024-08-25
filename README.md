<H3>ENTER YOUR NAME: SARAVANA KUMAR </H3>
<H3>ENTER YOUR REGISTER NO: 212222230133</H3>
<H3>EX. NO.1</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('Churn_Modelling.csv')
df.head()
df.isnull().sum()
df.duplicated().sum()
df=df.drop(['Surname', 'Geography','Gender'], axis=1)
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1.head()
x=df1.iloc[:,:-1]
y=df1.iloc[:,-1]
from sklearn.model_selection import train_test_split as t
xtrain,xtest,ytrain,ytest=t(x,y,test_size=0.2,random_state=0)
print("X-train:\n",xtrain,"\n---------------------------------------------------------------------------")
print("X-test:\n",xtest,"\n---------------------------------------------------------------------------")
print("Y-train:\n",ytrain,"\n---------------------------------------------------------------------------")
print("Y-test:\n",ytest,"\n---------------------------------------------------------------------------")
```

## OUTPUT:

### Load dataset:
![image](https://github.com/user-attachments/assets/29d73016-3cc8-4efc-af15-79d0b3822d98)

### Check for null values:
![image](https://github.com/user-attachments/assets/4b0e2319-f062-46dc-b969-91d95efd5974)

### Check for duplicate values:
![image](https://github.com/user-attachments/assets/f3bd221d-0adb-4ba7-b4fc-de6ec0e0921f)

### Dropping the unwanted data:
![image](https://github.com/user-attachments/assets/6ee0d962-fdc0-4e17-ab87-54efaf0410e0)

### Normalizing values:
![image](https://github.com/user-attachments/assets/fd74f7a6-36e0-44a0-8359-6f30ce2fc73f)

### Splitting the dataset for Model creation:
![image](https://github.com/user-attachments/assets/624165c5-f135-45e9-9382-890a3e0fe643)

### xtrain:
![image](https://github.com/user-attachments/assets/83636276-6a46-4ec3-add3-a54e7b9c2e9f)

### xtest:
![image](https://github.com/user-attachments/assets/87cb2892-ee9e-49d9-873c-7268e0eff828)

### ytrain:
![image](https://github.com/user-attachments/assets/9e29d0e2-56b3-4474-bac3-4e08dbf3a279)

### ytest:
![image](https://github.com/user-attachments/assets/11a0a6b2-7d29-494d-b115-bb1b39f2b14e)













## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


