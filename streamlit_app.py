# streamlit umumnya diinisialisasi dengan 'st'
#%%
import streamlit as st
import pandas as pd
import itertools
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

import time
import pickle

#%%

with open("hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)

df.replace(-9.0, np.NaN, inplace=True)

df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

df_selected.rename(columns=column_mapping, inplace=True)

columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

df_clean = df_selected.fillna(value=fill_values)
df_clean.drop_duplicates(inplace=True)

X = df_clean.drop("target", axis=1)
y = df_clean['target']

#%%
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

model = pickle.load(open("xgboost_oversample_model.pkl", 'rb'))


#%%
y_pred = model.predict(X_smote_resampled)
accuracy = accuracy_score(y_smote_resampled, y_pred)
accuracy = round((accuracy * 100))

df_final = X_smote_resampled
df_final['target'] = y_smote_resampled
#%%
#--------------------------------------------------------------------------------------------------------------------------------
# User Interface
st.set_page_config(
    page_title = "Predict Hungarian heart disease",
    page_icon = ':heart' #nama emoji 
)

# st.write dapat digunakan menampilkan test,dataframe,visualisasi
st.title('Predict Hungarian Heart Disease')
st.write('information about heart disease')
st.write("")
tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])
st.write("")
with tab1:
  st.header("User Input Features")


  age = st.number_input("Age", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  sex = st.selectbox("Sex", options=["Male", "Female"])
  if sex == "Male":
      sex = 1
  elif sex == "Female":
      sex = 0
    # -- Value 0: Female
    # -- Value 1: Male
    
  cp = st.selectbox("Chest pain type", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  if cp == "Typical angina":
      cp = 1
  elif cp == "Atypical angina":
    cp = 2
  elif cp == "Non-anginal pain":
    cp = 3
  elif cp == "Asymptomatic":
    cp = 4
    # -- Value 1: typical angina
    # -- Value 2: atypical angina
    # -- Value 3: non-anginal pain
    # -- Value 4: asymptomatic
    
  trestbps = st.number_input("resting blood pressure (in mm Hg on admission to the hospital)", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())

  chol = st.number_input("Serum cholestrol (in mg/dl)", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())

  fbs = st.selectbox("Fasting blood sugar > 120 mg/dl? ", options=["False", "True"])
  if fbs == "False":
    fbs = 0
  elif fbs == "True":
    fbs = 1
    # -- Value 0: false
    # -- Value 1: true
    
  restecg = st.selectbox("Resting electrocardiographic results", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  if restecg == "Normal":
    restecg = 0
  elif restecg == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  thalach = st.number_input("Maximum heart rate achieved", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())

  exang = st.selectbox("Exercise induced angina?", options=["No", "Yes"])
  if exang == "No":
    exang = 0
  elif exang == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
    
  predict_btn = st.button("Predict", type=("primary"))
  result = ":violet[-]"

  st.write("")
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]"

  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

  st.write(f"**_Model's Accuracy_** :  :green[**{accuracy}**]%")
    

# %%
import random

number = random.randint(5, 100)

with tab2:
  st.header("Predict multiple data:")

  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)

# %%
