
import streamlit as st 
import numpy as np
import pandas as pd
import joblib

# First lets load the instances that were created

with open ('scaler.joblib','rb') as file:
    scale = joblib.load(file)

with open ('pca.joblib','rb') as file:
    pca = joblib.load(file)

with open ('final_model.joblib','rb') as file:
    model = joblib.load(file)

def prediction(input_list):

    scaled_input = scale.transform([input_list])
    pca_input = pca.transform(scaled_input)
    output = model.predict(pca_input)[0]

    if output == 0:
        return 'underdeveloped'
    elif output == 1 :
        return 'Developed'
    else:
        return 'Developing'

def main():

    st.title('HELP NGO FOUNDATION')
    st.subheader('This application will give the status of the country based on socio-economic and health')

    gdp = st.text_input('Enter the GDP per Population of a country')
    inc = st.text_input('Enter the  per capita income of a country')
    imp = st.text_input('Enter the  Imports in terms of % of GDP ')
    exp = st.text_input('Enter the  Exports in terms of % of GDP ')
    inf = st.text_input('Enter the inflation rate in a country (%)')

    hel = st.text_input('Enter the expenditure on health in term % of GDP')
    ch_m = st.text_input('Enter the number of death per 1000 birth for <5 yrs')
    fer = st.text_input('Enter the avg children born to a women in a country')
    lf = st.text_input('Enter the average life expectancy in a counrty')

    in_data = [ch_m,exp,hel,imp,inc,inf,lf,fer,gdp]

    if st.button('Predict'):
        response = prediction(in_data)
        st.success(response)

if __name__=='__main__':
    main()

