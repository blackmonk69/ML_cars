
import streamlit as st
#from utils import PreProcesor, columns 

import numpy as np
import pandas as pd
import joblib
import pickle
import sklearn

model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')
st.title('Predicting Car Prices using ML and a Linear Regression Model')
names=sorted(car['name'].unique())
companies=sorted(car['company'].unique())
car_models=sorted(car['name'].unique())
years=sorted(car['year'].unique(),reverse=True)
fuel_types=car['fuel_type'].unique()
companies.insert(0,'Select Company')
# 'name', 'company', 'year', 'kms_driven', 'fuel_type'

names=st.selectbox("choose name",names)
company = st.selectbox("Choose company",companies)
company="Audi"
car_model  = st.selectbox("Choose Car Model", car_models)
year = st.selectbox("Choose year", years)
fuel_type= st.selectbox("Choose fuel type",fuel_types)
driven=st.text_input ('km driven')
driven=3455
prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
print(prediction)
st.write(f"la prediccion es {prediction}")
#trigger = st.button('Predict', on_click=predict)