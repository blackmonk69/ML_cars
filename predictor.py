import streamlit as st
import numpy as np
import pandas as pd
import pickle
# let us load the trained model
model=pickle.load(open('LinearRegressionModel.pkl','rb'))

car=pd.read_csv('Cleaned_Car_data.csv')
st.title('Predicting Car Prices using ML - Linear Regression Model')
#these lists help us populate the select boxes
names=sorted(car['name'].unique())
companies=sorted(car['company'].unique())
car_models=sorted(car['name'].unique())
years=sorted(car['year'].unique(),reverse=True)
fuel_types=car['fuel_type'].unique()
# 'name', 'company', 'year', 'kms_driven', 'fuel_type'

names=st.selectbox("choose name",names)
company = st.selectbox("Choose company",companies)
car_model  = st.selectbox("Choose Car Model", car_models)
year = st.selectbox("Choose year", years)
fuel_type= st.selectbox("Choose fuel type",fuel_types)
driven = st.text_input ('km driven')

def predict(car_model,company,year,driven,fuel_type):
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    return round(prediction[0])
    
if st.button("Predict value"):
    prediccion='{:,}'.format(predict (car_model,company,year,driven,fuel_type))
    st.subheader (f"la prediccion es: $ {prediccion}")
    

