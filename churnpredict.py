undefinedFirst, let’s load the dataframe into Python with the pandas library and take a look at its head. I’ve renamed the file to “customer_churn.csv”, and it is the name I will be using below:
import pandas as pd

df = pd.read_csv('Customer_Churn.csv')
df.head()


undefinedLet’s look into these variables further by listing them out:
df.info()

undefinedLet’s count the number of customers in the dataset who have churned: 
df["Churn"].value_counts()

undefinedWe will start by analyzing the demographic data points:
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

cols = ['gender','SeniorCitizen',"Partner","Dependents"]
numerical = cols

plt.figure(figsize=(20,4))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sns.countplot(x=str(col), data=df)
    ax.set_title(f"{col}")


undefinedNow, let’s look into the relationship between cost and customer churn. In the real world, users tend to unsubscribe to their mobile service provider and switch to a different brand if they find the 
monthly subscription cost too high. Let’s check if that behavior is reflected in our dataset:
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

undefinedFinally, let’s analyze the relationship between customer churn and a few other categorical variables captured in the dataset:
cols = ['InternetService',"TechSupport","OnlineBackup","Contract"]

plt.figure(figsize=(14,4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x ="Churn", hue = str(col), data = df)
    ax.set_title(f"{col}")


undefinedNotice that the variable “TotalCharges” has the data type “object,” when it should be a numeric column. Let’s convert this column into a numeric one:
df['TotalCharges'] = df['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

undefinedFirst, let’s take a look at the categorical features in the dataset:
cat_features = df.drop(['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure'],axis=1)

cat_features.head()


undefinedNow, let’s take a look at the dataset after encoding these categorical variables:
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)
df_cat.head()


undefinedFinally, run the following lines of code to merge the dataframe we just created with the previous one:
num_features = df[['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure']]
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)


undefinedBefore we oversample, let’s do a train-test split. We will oversample solely on the training dataset, as the test dataset must be representative of the true population:
from sklearn.model_selection import train_test_split

finaldf = finaldf.dropna()
finaldf = finaldf.drop(['customerID'],axis=1)

X = finaldf.drop(['Churn'],axis=1)
y = finaldf['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


undefinedNow, let’s oversample the training dataset:
from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote


undefinedLet’s check the number of samples in each class to ensure that they are equal:
y_train.value_counts()

undefinedWe will now build a random forest classifier to predict customer churn:
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)


undefinedLet’s evaluate the model predictions on the test dataset:
from sklearn.metrics import accuracy_score

preds = rf.predict(X_test)
print(accuracy_score(preds,y_test))
