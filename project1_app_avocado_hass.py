# Import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle
# evaluation libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score

# funcsion libraries
import sys
import os
import re

import import_ipynb
from lib.function_lib import *

model_pre = pickle.load(open('model/Problem1_model_pre_processing.sav', 'rb'))
model_scaler = pickle.load(open('model/Problem1_model_standardizing.sav', 'rb'))
model_predict = pickle.load(open('model/Problem1_ex_model.sav', 'rb'))

# GUI
st.sidebar.markdown("<h1 style='text-align: left; color: Blue;'>CATALOG</h1>", unsafe_allow_html=True)
menu = ["Introduction","Summary about Projects", 'Predict prices by Regression', "Predict prices by Time series", "Conclusion and Next steps"]
choice = st.sidebar.radio("Choose one of object",menu)

# 1. Introduction
if choice == 'Introduction':
    st.header('INTRODUCTION')

# 2. Summary about Projects
elif choice == 'Summary about Projects':
    st.header('SUMMARY ABOUT PROJECTS')

# 3. Predict prices by Regression
elif choice == 'Predict prices by Regression':


    # content
    st.header('PREDICT PRICES BY REGRESSION')

    # sidebar
    menu3_input = ['Input data','Load file']
    choice3_input = st.sidebar.selectbox('Choose the way to input data', menu3_input)

    menu3_model = ['ExtraTreesRegressor','RandomForestRefressor','BaggingRegressor','KNeighborsRegressor','ANN']
    choice3_model = st.sidebar.selectbox('Choose the model', menu3_model)


    if choice3_input =='Input data':
        # sidebar - input
        fea_type = st.sidebar.radio("Type",['conventional','organic'])
        fea_region = st.sidebar.selectbox("Region",['region1','region2'])
        fea_PLU_4046 = st.sidebar.number_input("Volume of PLU 4046", value = 1)
        fea_PLU_4225 = st.sidebar.number_input("Volume of PLU 4225", value = 1)
        fea_PLU_4770 = st.sidebar.number_input("Volume of PLU 4770", value = 1)
        fea_Total_Volume = st.sidebar.number_input("Total volume", value = 1)
        fea_Small_Bags = st.sidebar.number_input("Number of Small bags", value = 1)
        fea_Large_Bags = st.sidebar.number_input("Number of Large bags", value = 1)
        fea_XLarge_Bags = st.sidebar.number_input("Number of XLarge bags", value = 1)
        fea_Total_Bags = st.sidebar.number_input("Total bags", value = 1)

        # show results
        if st.sidebar.button("Show predict results"):
        
        # content
            if choice3_model == 'ExtraTreesRegressor':
                st.markdown('ExtraTreesRegressor')            


            if choice3_model == 'RandomForestRefressor':
                st.markdown('RandomForestRefressor')  


            if choice3_model == 'BaggingRegressor':
                st.markdown('BaggingRegressor')  


            if choice3_model == 'KNeighborsRegressor':
                st.markdown('KNeighborsRegressor')  



            if choice3_model == 'ANN':
                st.markdown('ANN')  


    elif choice3_input == 'Load file':
        # sidebar
        try:
            uploaded_file = st.sidebar.file_uploader('Upload data', type = ['csv'])
            dir_file = 'data/' + uploaded_file.name

        except Exception as failGeneral:        
            print("Fail system, please call developer...", type(failGeneral).__name__)
            print("Description:", failGeneral)
        finally:
            print("File uploaded")       

        # prediction
        data = pd.read_csv(dir_file)
        X_new, y_new = model_pre.transform(data)
        X_scale = model_scaler.transform(X_new)
        yhat = model_predict.predict(X_scale)
        r2_score = r2_score(y_new, yhat)

        # show results
        if st.sidebar.button("Show predict results"):

            st.write("yhat")
            st.dataframe(yhat)
            st.markdown('r2_score: ' + str(r2_score))


# 4. Predict prices by Time series
elif choice == 'Predict prices by Time series':
    # content
    st.header('PREDICT PRICES BY TIME SERIES')

    # sidebar - input
    fea_month = st.sidebar.number_input("Choose the length of time", value = 1)


    # show results
    if st.sidebar.button("Show predict results"):
    
    # content
        st.markdown('asd')



# 5. Conclusion and Next steps
elif choice == 'Conclusion and Next steps':
    st.header('CONCLUSION AND NEXT STEPS')
