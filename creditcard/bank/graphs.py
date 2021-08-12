import io
import urllib
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def get_graph():
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri

# Read the data file
CC_data = pd.read_csv(r'C:\Users\Ruchita Potamsetti\SdlProject\creditcard\bank\CC_data.csv', encoding='ISO-8859-1')
#CC_data.shape
CC_data['Approved'].replace('+','yes',inplace=True)
CC_data['Approved'].replace('-','no',inplace=True)
CC_data['PriorDefault'].replace('f','yes',inplace=True)
CC_data['PriorDefault'].replace('t','no',inplace=True)
CC_data.rename(columns={'Male': 'Gender'},inplace=True)
CC_data['Gender'].replace('b','Male',inplace=True)
CC_data['Gender'].replace('a','Female',inplace=True)
CC_data2 = CC_data.copy()

#DATA PREPROCESSING

# Replace "?" with NaN
CC_data.replace('?', np.NaN, inplace = True)
# Convert Age to numeric
CC_data["Age"] = pd.to_numeric(CC_data["Age"])
#CC_data.isnull().sum()
# Imputing missing values for numerical columns with mean value
CC_data.fillna(CC_data.mean(), inplace=True)
#CC_data.isnull().sum()


#ANALYZING DATA
    
def gender():
    sns.countplot(x = 'Gender',hue = 'Approved' ,data = CC_data)
    plt.title('Gender & Approved')

def prior():
    sns.catplot(x="Gender", hue="PriorDefault", col="Approved",
                data= CC_data, kind="count",
                height=5, aspect=.7)

# def education():
#     sns.countplot(x="EducationLevel", hue="Approved", data=CC_data)
#     plt.title('Education Level')

def prediction1():
    CC_data = pd.read_csv(r'C:\Users\Ruchita Potamsetti\SdlProject\creditcard\bank\CC_data.csv', encoding='ISO-8859-1')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # # Looping for each object type column
    # Using label encoder to convert into numeric types
    for col in CC_data:
        if CC_data[col].dtypes == 'object':
            CC_data[col] = le.fit_transform(CC_data[col])

    CC_data = CC_data.drop(['DriversLicense', 'ZipCode'], axis=1)
    CC_data = CC_data.values

    X, y = CC_data[:, 0:13], CC_data[:, 13]

    # Spliting the data into training and testing sets
    X_train, X_test, y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=123)

    # Scaling X_train and X_test
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scaler.fit_transform(X_train)
    rescaledX_test = scaler.transform(X_test)
    rescaledX = scaler.transform(X)

    # Import LogisticRegression
    from sklearn.linear_model import LogisticRegression

    # Fitting logistic regression with default parameter values
    logreg = LogisticRegression()
    logreg.fit(rescaledX_train, y_train)

    # Getting the accuracy score of predictive model
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(rescaledX_train, y_train)
    y_pred = rf.predict(rescaledX_test)
    # Evaluate the confusion_matrix
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix for RandomForest classifier')

