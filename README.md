# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
### STEP 1:Read the given Data.
### STEP 2:Clean the Data Set using Data Cleaning Process.
### STEP 3:Apply Feature Scaling for the feature in the data set.
### STEP 4:Apply Feature Selection for the feature in the data set.
### STEP 5:Save the data to the file.

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
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/b597a523-af47-48c1-828d-95810bd2e5c1)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/716a73e3-8fb1-43db-ad6e-f2cea7b5bafe)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/a2a7c822-2337-49b4-a388-6c340a7ea9d3)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=pd.read_csv("bmi.csv")
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/7ca6f945-f85a-462b-acc5-313697183b23)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/86b4aa59-a63f-4aea-a14b-456d90a864be)
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/c5a7f5b8-f034-4fca-b2ea-e2d59ee18199)
```
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/9c5e24b1-a146-44aa-9da4-7e6fcafc1415)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/edec9bee-7c04-4af0-852f-0871bb3e6c7e)
```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/45043a49-bea1-42de-9359-126a27408f1c)
```
df=pd.read_csv("income.csv")
df.info()
```
![image](https://github.com/user-attachments/assets/5d096efd-f7b7-4bad-afe9-6046b4f28557)
```
df.head()
```
![image](https://github.com/user-attachments/assets/bff55ef6-1772-4eec-bd7a-e7657421d9ef)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/986647a7-ed7f-49d7-a5c0-4ff651485ccf)
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/42e1c2a2-9164-4948-b1f8-0a5a8a103a44)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/a1007116-fdc7-4696-98f0-4b2f6ce3319c)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/07e7bbe1-acf3-4df4-895e-295cceb78930)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/ee792b93-1674-47d1-845a-8b3a98e18e92)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/fae5bcb8-4c37-4645-8a4e-5ad086140268)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/142bff04-a66d-45ea-a074-8dcf75d4a58f)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/399f692f-fbd7-4c33-b640-21282cb5aff7)
```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/5419fc83-72e9-4ad8-b5c2-57d0defc154d)
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/99f6bf08-b32c-4735-846b-604dfc8ccefa)

# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
