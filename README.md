# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
```
```
df=pd.read_csv("bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/a66652f9-c881-4a86-9d60-f92810b15c79)
```
df.head()
```
![image](https://github.com/user-attachments/assets/da322bb8-5154-4a88-956d-6b2ddc277842)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/be732e5d-f3d4-43d3-98cd-e3880852c0f4)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/f637300f-cced-442c-a903-39645a36dbfb)
```
from sklearn.preprocessing import MinMaxScaler
```
```
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])fr
```
```
df.head(10)
```
![image](https://github.com/user-attachments/assets/1365399a-aa4c-4cb7-a4b6-796831621969)

```
df1=pd.read_csv("bmi.csv")
```
```
df2=pd.read_csv("bmi.csv")
```
```
df3=pd.read_csv("bmi.csv")
```
```
df4=pd.read_csv("bmi.csv")
```
```
df5=pd.read_csv("bmi.csv")
```
```
df1

```

![image](https://github.com/user-attachments/assets/a7322a98-3297-41b6-a5e8-595c91dc8b4d)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![image](https://github.com/user-attachments/assets/6ce0dc2e-72ee-4283-ab1a-61e105b7b596)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
```
```
df2
```

![image](https://github.com/user-attachments/assets/d5705138-76ec-4a7f-ab0a-04f1e473fdc6)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df3
```

![image](https://github.com/user-attachments/assets/074d8e0c-80fa-47c0-bac6-7353f1b39ab7)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df4
```

![image](https://github.com/user-attachments/assets/de4deeba-b8ab-4c5d-8818-6768a02c8543)
```
import seaborn as sns
```
```
feature selection 
import pandas as pd

import numpy as np 
import seaborn as sns
```
```
import seaborn as sns
```
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
```
```
data=pd.read_csv("titanic_dataset.csv")
data
```
![image](https://github.com/user-attachments/assets/c0330f51-a459-4e63-ba8f-8356dc929c69)
```
data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
```
```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
```
```
data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes
```
```
 data
```
![image](https://github.com/user-attachments/assets/2a598b08-3121-478e-a22b-c7c0986f79c3)
```

k=5
selector=SelectKBest(score_func=chi2, k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
```
```
x_encoded =pd.get_dummies(x)
selector=SelectKBest(score_func=chi2, k=5)
x_new = selector.fit_transform(x_encoded,y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected_Feature:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/317c121c-abf4-4dc7-a14d-0f3e899e50b4)
```
selector=SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/70e1b6b3-e6a9-4b0c-abf6-e257e8dff14f)
```
selector=SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/1814553c-c56f-42a4-9416-589884be81bd)
 
```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/412dcc3f-3af8-4678-9da0-1a354173f174)
```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.1
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/afe8c286-877d-41f4-9392-3aa16803f600)
```
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.15
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/dca07307-d7fc-4546-8614-fadbeb133ca0)

# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
