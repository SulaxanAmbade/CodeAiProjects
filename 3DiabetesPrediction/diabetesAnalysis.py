import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Descriptive Analytics
st.title('Diabetes Dataset Analysis')

st.header('Descriptive Analytics')
st.write(df.describe())

# Binary feature distributions
st.subheader('Binary Feature Distributions')
for column in ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck']:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=column, width=0.3, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Continuous feature distributions
st.subheader('Continuous Feature Distributions')
for column in ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, bins=20, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Correlation with Diabetes
st.subheader('Correlation with Diabetes')
corrDf = df.drop('Diabetes_binary', axis=1).corrwith(df.Diabetes_binary)
fig, ax = plt.subplots()
ax.bar(corrDf.index, corrDf)
ax.set_title('Correlation with Diabetes')
ax.set_xlabel('Features')
ax.set_ylabel('Correlation')
plt.xticks(rotation=90)
plt.grid()
st.pyplot(fig)

# Correlation matrix
st.subheader('Correlation Matrix')
corMat = df.corr()
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corMat, cmap='viridis', annot=True, ax=ax)
plt.title('Correlation Matrix')
st.pyplot(fig)

# Predictive Analytics
X = df[['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
        'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']]
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conMat = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred)

st.header('Predictive Analytics')
st.write(f'Accuracy: {accuracy}')
st.write('Classification Report:')
st.text(classReport)

st.subheader('Confusion Matrix')
fig, ax = plt.subplots()
sns.heatmap(conMat, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Prescriptive Analytics
def GetUserData():
    user_data = {}
    user_data['HighBP'] = st.number_input("High Blood Pressure (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['HighChol'] = st.number_input("High Cholesterol (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['BMI'] = st.number_input("Body Mass Index (e.g., 25.3): ", min_value=0.0)
    user_data['Smoker'] = st.number_input("Smoker (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['Stroke'] = st.number_input("History of Stroke (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['HeartDiseaseorAttack'] = st.number_input("Heart Disease or Attack (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['PhysActivity'] = st.number_input("Physical Activity (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['Fruits'] = st.number_input("Fruit Consumption (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['Veggies'] = st.number_input("Vegetable Consumption (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['HvyAlcoholConsump'] = st.number_input("Heavy Alcohol Consumption (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['GenHlth'] = st.number_input("General Health (1-5, with 1 being Excellent and 5 being Poor): ", min_value=1, max_value=5)
    user_data['MentHlth'] = st.number_input("Mental Health (Number of days in the past 30 days your mental health was not good): ", min_value=0)
    user_data['PhysHlth'] = st.number_input("Physical Health (Number of days in the past 30 days your physical health was not good): ", min_value=0)
    user_data['DiffWalk'] = st.number_input("Difficulty Walking (1 for Yes, 0 for No): ", min_value=0, max_value=1)
    user_data['Sex'] = st.number_input("Sex (1 for Male, 0 for Female): ", min_value=0, max_value=1)
    user_data['Age'] = st.number_input("Age: ", min_value=0)
    return pd.DataFrame([user_data])

st.header('Prescriptive Analytics')
userData = GetUserData()
udScaled = scaler.transform(userData)
prediction = model.predict(udScaled)
predictProbability = model.predict_proba(udScaled)

if prediction[0] == 1:
    st.write("The model predicts that you have diabetes.")
else:
    st.write("The model predicts that you do not have diabetes.")

probability_no_diabetes = predictProbability[0][0]
probability_diabetes = predictProbability[0][1]

st.write(f'Probability of not having diabetes: {probability_no_diabetes:.2f}')
st.write(f'Probability of having diabetes: {probability_diabetes:.2f}')
