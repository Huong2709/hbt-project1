# Import library
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
import pickle
# data pre-processing libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
## tensorflow libraries to create ANN and build model
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, Sequential, Input, Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import load_model
# Facebook prophet
from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot
from math import sqrt

# evaluation libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score

# funcsion libraries
import sys
import os
import re

import import_ipynb
from lib.function_lib_v2 import *

raw_data = pd.read_csv('data/avocado.csv', index_col=0)
raw_data.columns = raw_data.columns.str.replace(" ","_")
raw_data.columns = ['Date', 'AveragePrice', 'Total_Volume', 'PLU_4046','PLU_4225','PLU_4770',
       'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 'type', 'year',
       'region']
lst_region = raw_data.loc[raw_data['region'] != "TotalUS", 'region'].unique()
lst_cate = pd.read_csv('model/lst_cate.csv').iloc[:,0].to_list()
lst_cont = pd.read_csv('model/lst_cont.csv').iloc[:,0].to_list()


# GUI
st.sidebar.markdown("<h1 style='text-align: left; color: Black;'>CATALOG</h1>", unsafe_allow_html=True)
menu = ["Introduction","Summary about Projects", 'Predict prices by Regression', "Predict prices by Time series", "Conclusion and Next steps"]
choice = st.sidebar.radio("Choose one of objects below", menu)

st.write("<h1 style='text-align: left; color: Red; font-size:40px'>AVOCADO HASS PRICE PREDICTION</h1>", unsafe_allow_html=True)


# 1. Introduction
if choice == 'Introduction':
    st.header('INTRODUCTION')
    st.write("**According to some researchs, there are a key market trend: Increasing Awareness Regarding the Health Benefits of Hass Avocado.**")
    st.write('''Avocados are a nutrient-dense food that provides relatively many health 
                benefits that include abundant nutrients (vitamins C, B5, B6, E, and K), potassium, 
                folate, protein, and healthy fats. Seven of the nine grams of carbohydrates in a 100-gram 
                serving are fiber, qualifying avocados as a low-carb food. The daily consumption of Hass 
                avocado as part of a hypocaloric diet supports weight loss and helps to increase gut bacteria. 
                An increased interest in plant-based and low-carbohydrate clean eating diets, such as the 
                keto diet and paleo diet, also contributed to the increased demand for Hass avocados.''')
    st.image("material/Intro_hass_avocado.jpg")
    st.write('''Avocado has become a favorite ingredient in many dishes (guacamole dip, salads, and sandwiches) 
                and is now on the menu of several restaurants across the United States. The Avocado toast, 
                a humble breakfast that was earlier eaten mainly in California and Australia, 
                is now a trendy menu item in many restaurants around the world. According to the USDA, 
                US demand for avocados has increased steadily over the past two decades.''')
    st.write('''To keep up with this trend and expand business in the right regions, 
                growing the right type of avocados, companies will need to predict the future prices 
                of avocados in each region. Based on the predicted price, combine many 
                other factors to make decisions and expand business in time with market trends.''')


# 2. Summary about Projects
elif choice == 'Summary about Projects':
    st.header('SUMMARY ABOUT PROJECTS')
    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Business Understanding</h3>", unsafe_allow_html=True)
    st.write("**Current status:**")
    st.write('''The company trades 'Hass' avocados butter in many regions of 
                United States with 2 types of butter: conventional butter and organic butter
                Packed according to many standards (Small/Large/XLarge Bags),
                and have 3 different PLUs (Product Look Up) (4046, 4225,
                4770). But they don't have a model to predict avocado prices for yet
                expansion.''')
    st.write("**Objective/Problem:**")
    st.write('''Build a model for average price prediction
                average of "Hass" avocados in the US. This model
                will help the company to consider expanding production and business.''')

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Understanding/Acquire</h3>", unsafe_allow_html=True)
    st.write("- Data is taken directly from retailers' cash registers based on actual retail sales of Hass avocados.")
    st.write("- The data represent data taken from the weekly retail scanner for retail volume (National retail volume- units) and avocado prices from April 2015 to March 2018.")
    st.write("- The Average Price in the table reflects the price per unit (per avocado), even when multiple units (avocados) are sold in the bag.")
    st.write("- Product Lookup codes (PLU's) in the table are for Hass butter only, not for other products.")
    st.write("- All data is dumped and stored in the file avocado.csv with 18249 records. With columns:")
    st.write("&nbsp;"*9 + "- Date: The date of the observation")
    st.write("&nbsp;"*9 + "- AveragePrice: The average price of a single avocado")
    st.write("&nbsp;"*9 + "- Total Volume: Total number of avocados sold")
    st.write("&nbsp;"*9 + "- 4046: Total number of avocados with PLU 4046 sold")
    st.write("&nbsp;"*9 + "- 4225: Total number of avocados with PLU 4225 sold")
    st.write("&nbsp;"*9 + "- 4770: Total number of avocados with PLU 4770 sold")
    st.write("&nbsp;"*9 + "- Total Bags: Total number of bags sold")
    st.write("&nbsp;"*9 + "- Small Bags: Total number of Small bags sold")
    st.write("&nbsp;"*9 + "- Large Bags: Total number of Large bags sold")
    st.write("&nbsp;"*9 + "- XLarge Bags: Total number of XLarge bags sold")
    st.write("&nbsp;"*9 + "- year: The year sold")
    st.write("&nbsp;"*9 + "- region: The city or region of the observation")
    st.write("- There are two types of avocados in the dataset and several different regions. This allows us to do all sorts of analyzes for different regions, or analyze the entire US for either type of avocado.")

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>First 5 rows of dataset:</h6>", unsafe_allow_html=True)
    st.dataframe(raw_data.head())

    st.write("**Statistics information:**")
    st.write(continous_feature_describe(lst_cont, df=raw_data))
    st.write("There are a large range between min and max values of these variables: Total_Volume, PLU_4046, PLU_4225, PLU_4770, Total_Bags, Small_Bags, Large_Bags, XLarge_Bags. => We need to do deeper analysis on these variables.")

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>Density chart:</h6>", unsafe_allow_html=True)
    st.image('material/EDA_displot.jpg')

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>Correlation matrix:</h6>", unsafe_allow_html=True)
    st.dataframe(raw_data[lst_cont].corr())

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>AveragePrice by type:</h6>", unsafe_allow_html=True)
    st.image('material/EDA_price_by_type.jpg')

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>AveragePrice by date:</h6>", unsafe_allow_html=True)
    st.image('material/EDA_price_by_date.jpg')

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>AveragePrice by region:</h6>", unsafe_allow_html=True)
    st.image('material/EDA_price_by_region.jpg')

    st.write("<h1 style='text-align: left; color: Black; font-size:25px'>Objective 1: Using Regression Algorithms to predict AveragePrice</h1>", unsafe_allow_html=True)
    
    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Preparation/Acquire</h3>", unsafe_allow_html=True)
    st.write("1. Create X and y.")
    st.write("2. Scale continous variables by RobustScaler. Because continous variables don't have normal distribution and have outliers.")
    st.write("3. Create 2 dataset: scaled and normal to build and evaluate models.")

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Modeling and Evaluation/Acquire</h3>", unsafe_allow_html=True)
    st.write("1. Select general model by LazyPredict and by a defination function. There are serveral models that have high performance: ExtraTreesRegressor, RandomForestRefressor, BaggingRegressor, KNeighborsRegressor.")
    st.dataframe(pd.read_csv('model/Problem1_compare_models.csv').iloc[:,1:])
    st.write("2. Build each models one-by-one.")
    st.write("3. Try an network named ANN.")
    st.write("4. Both have a good performance and be saved for using.")
    st.write("5. The best model is ExtraTreesRegressor.")
    st.write("Evaluation metrics:")
    st.write("&nbsp;"*9 + "- Train Dataset RMSE:  1.8455e-06")
    st.write("&nbsp;"*9 + "- Test Dataset RMSE:  0.1520")
    st.write("&nbsp;"*9 + "- Train Dataset R-square:  0.9999")
    st.write("&nbsp;"*9 + "- Test Dataset R-square:  0.8548")
    st.write("Evaluation visualization - train dataset")
    st.image('material/P1_ExtraTree_train.jpg')
    st.write("Evaluation visualization - test dataset")
    st.image('material/P1_ExtraTree_test.jpg')
    st.write("ExtraTreesRegressor and several other models have high performance (r2-score) and RMSE < standard deviation of original dataset.")

    st.write("<h1 style='text-align: left; color: Black; font-size:25px'>Objective 2: Using Time Series Algorithms to forecast AveragePrice of specific type of avocado and region</h1>", unsafe_allow_html=True)

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Preparation/Acquire</h3>", unsafe_allow_html=True)
    st.write("1. Pre-processing data to create a dataframe with 2 columns: AveragePrice and Month (format as YYYY-MM-01). There are 39 samples, not much.")
    st.write("2. Do decompostion and find out: AveragePrice has a very clear month-to-month seasonality. Uptrend line is from 2016. So we could use historical AveragePrice to forecast for future.")

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Modeling and Evaluation/Acquire</h3>", unsafe_allow_html=True)
    st.write("1. For each type = organic/conventional and each specific region, build some models and evaluation these models:")
    st.write("&nbsp;"*9 + "- ARIMA: not good.")
    st.write("&nbsp;"*9 + "- Holt-Winter and RNN LSTM: Some combination of type (conventional/organic) and region has good performance. RMSEs is less than the standard deviation of the original data set. But when forecasting for the future, the results don't make sense. Probably because the number of samples to train the model is quite small.")
    st.write("&nbsp;"*9 + "- Facebook Prophet: Some combination of type (conventional/organic) and region has good performance. RMSEs is less than the standard deviation of the original data set. The forecasting result alse makes sense.")
    st.dataframe(pd.read_csv('model/Problem4_organic_region_assessment_df.csv').iloc[:,1:])
    st.dataframe(pd.read_csv('model/Problem4_conventional_region_assessment_df.csv').iloc[:,1:])
    st.write("2. The good models will be use to forecast future.")

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Suggestion</h3>", unsafe_allow_html=True)
    st.write("Some assumptions are made to support the methodology:")
    st.write("1. Average price always cover the cost.")
    st.write("2. The boss is interested in margin. When determining the desired margin, the regression model will be used to determine the volume to be produced.")
    st.write("We will choose regions and type = organic/conventional to expand base on the amplitudes margin.")
    st.write("*Notes: Margin (5-years) = min trend - max trend (meaning the average price rises faster)*")




#####################################################################################################
# PROJECT 1 - REGRESSION
#####################################################################################################

# 3. Predict prices by Regression
elif choice == 'Predict prices by Regression':

#----------------------------------------------------------------------------------------------------
    # 3.1. pre_processing
    # if __name__=='__main__':
    #     model_pre = pickle.load(open('model/Problem1_model_pre_processing.sav', 'rb'))
    #     model_scaler = pickle.load(open('model/Problem1_model_standardizing.sav', 'rb'))
        # model_ex = pickle.load(open('model/Problem1_ex_model.sav', 'rb'))
        # model_rf = pickle.load(open('model/Problem1_rf_model.sav', 'rb'))
        # model_bg = pickle.load(open('model/Problem1_bg_model.sav', 'rb'))
        # model_kn = pickle.load(open('model/Problem1_kn_model.sav', 'rb'))
        # model_ann = load_model('model/Problem1_ANN_model.h5')

    def pre_processing(data):
        import pickle
        # load list categorical & continous variables
        lst_cate = pd.read_csv('model/lst_cate.csv').iloc[:,0].to_list()
        lst_cont = pd.read_csv('model/lst_cont.csv').iloc[:,0].to_list()

        # load encoder model
        encoder = pickle.load(open('model/OneHotEncoder.sav','rb'))

        # data cleaning
        df = data.copy()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # drop duplicates
        df = df.drop_duplicates() 

        ## change datatype
        # df['Date'] = pd.to_datetime(df['Date']) 

        ## change column names
        df.columns = df.columns.str.replace(" ","_")
        if 'AveragePrice' in df.columns:
            df.columns = ['Date', 'AveragePrice', 'Total_Volume', 'PLU_4046','PLU_4225','PLU_4770',
              'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 'type', 'year',
              'region']
        else: 
            pass
        # remove some region: 
        df = df.loc[df['region'] != "TotalUS",:]

        # apply encoder on categorial columns
        lst_encode = ['type','region']

        # arr = encoder.transform(df[lst_encode]).toarray()
        arr = encoder.transform(df[lst_encode])

        cols = []
        n = 0
        for i in encoder.categories_:
            for j in i[0:]: 
                t = 'oh_' + lst_encode[n] + '_' +str(j)
                t = t.replace('-', '_')
                cols.append(t)
            n = n+1
        df_oh_encode = pd.DataFrame(arr, columns=cols)
        
        # concat encoded columns to df
        lst_concat = lst_cont
        df = pd.concat([df_oh_encode.reset_index(), df.reset_index()], axis=1).iloc[:,1:]

        # create X, y
        lst_cont_fea = lst_cont
        lst_cont_fea.remove('AveragePrice')        
        X = pd.concat([df.loc[:, df.columns.str.startswith("oh_")], \
               df[lst_cont_fea]], axis=1)
        
        if 'AveragePrice' in df.columns:
            y = df['AveragePrice']
        else: 
            y=[]

        return X, y
    
    # 3.2. scaler
    def scaling(data):
        import pickle
        # load scaler model
        scaler = pickle.load(open('model/Problem1_RobustScaler.sav','rb'))

        # transform data
        lst_cont = pd.read_csv('model/lst_cont.csv').iloc[:,0].to_list()
        lst_cont_fea = lst_cont
        lst_cont_fea.remove('AveragePrice')   

        X = data.copy()
        X_before_scale = X[lst_cont_fea]
        X_scale = scaler.transform(X_before_scale) #--> dùng transform, không dùng fit_transform cho tập Test
        X_scale = pd.DataFrame(X_scale, columns=(X_before_scale.add_suffix('_scale')).columns)

        # concat data
        X_new = pd.concat([X.reset_index(drop=True), X_scale], \
                                axis=1)
        # select scaled features
        X_scale = pd.concat([X_new.loc[:, X_new.columns.str.startswith("oh_")], \
                                    X_new.loc[:, X_new.columns.str.endswith("_scale")]], \
                                    axis=1)
      
        return X_scale

    # 3.2. build model
    ## Load data
    X_train_scale = pd.read_csv('model/X_train_scale.csv').iloc[:,1:]
    y_train = pd.read_csv('model/y_train.csv').iloc[:,1:]
    X_test_scale = pd.read_csv('model/X_test_scale.csv').iloc[:,1:]
    y_test = pd.read_csv('model/y_test.csv').iloc[:,1:]

#----------------------------------------------------------------------------------------------------

    # content
    st.header('PREDICT PRICES BY REGRESSION')

    # sidebar
    menu3_input = ['Input data','Load file']
    choice3_input = st.sidebar.selectbox('Choose the way to input data', menu3_input)

    # menu3_model = ['ExtraTreesRegressor','RandomForestRefressor','BaggingRegressor','KNeighborsRegressor','ANN']
    menu3_model = ['ExtraTreesRegressor','RandomForestRefressor','BaggingRegressor']
    choice3_model = st.sidebar.selectbox('Choose the model', menu3_model)


    if choice3_input =='Input data':
        # sidebar - input
        fea_type = st.sidebar.radio("Type",['conventional','organic'])
        fea_region = st.sidebar.selectbox("Region", lst_region)
        fea_PLU_4046 = st.sidebar.number_input("Volume of PLU 4046", value = 1.00, format="%.2f", step=0.01)
        fea_PLU_4225 = st.sidebar.number_input("Volume of PLU 4225", value = 1.00, format="%.2f", step=0.01)
        fea_PLU_4770 = st.sidebar.number_input("Volume of PLU 4770", value = 1.00, format="%.2f", step=0.01)
        fea_Total_Volume = st.sidebar.number_input("Total volume", value = 1.00, format="%.2f", step=0.01)
        fea_Small_Bags = st.sidebar.number_input("Number of Small bags", value = 1.00, format="%.2f", step=0.01)
        fea_Large_Bags = st.sidebar.number_input("Number of Large bags", value = 1.00, format="%.2f", step=0.01)
        fea_XLarge_Bags = st.sidebar.number_input("Number of XLarge bags", value = 1.00, format="%.2f", step=0.01)
        fea_Total_Bags = st.sidebar.number_input("Total bags", value = 1.00, format="%.2f", step=0.01)

        # show results
        if st.sidebar.button("Show predict results"):
            lst_input = [fea_type, fea_region, fea_PLU_4046, fea_PLU_4225, fea_PLU_4770, fea_Total_Volume, fea_Small_Bags, fea_Large_Bags, fea_XLarge_Bags, fea_Total_Bags]
            col_names = ['type', 'region', 'PLU_4046', 'PLU_4225', 'PLU_4770', 'Total Volume', 'Small Bags', 'Large Bags', 'XLarge Bags', 'Total Bags']
            input_data = pd.DataFrame(lst_input)
            input_data = input_data.T
            input_data.columns = col_names
            # X_new, y_new = model_pre.transform(input_data)
            # X_scale = model_scaler.transform(X_new)  
            X_new, y_new = pre_processing(input_data)
            X_scale = scaling(X_new)       
        # content
            if choice3_model == 'ExtraTreesRegressor':
                # ExtraTreesRegressor (build again, because saved model > 800MB)
                model_ex = ExtraTreesRegressor(n_estimators=50, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            max_features='sqrt', 
                                            max_depth=None, 
                                            bootstrap=False)
                model_ex = model_ex.fit(X_train_scale, y_train)              
                # prediction
                model_predict = model_ex
                yhat = model_predict.predict(X_scale)                

                # show results
                st.markdown("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: ExtraTreesRegressor")
                st.markdown("2. Input data: ")
                for i in input_data.columns[:-3]:
                    st.write('&nbsp;'*10 + '- ' + i + ': ' + str(input_data[i][0]))
                # st.dataframe(input_data)
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.write("Prediction price: " + str(round(yhat[0],3)))                          


            elif choice3_model == 'RandomForestRefressor':
                # RandomForestRegressor (build again, because saved model > 800MB)  
                model_rf = RandomForestRegressor(n_estimators=50, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            max_features='sqrt', 
                                            max_depth=None, 
                                            bootstrap=False)
                model_rf = model_rf.fit(X_train_scale, y_train)
                # prediction
                model_predict = model_rf
                yhat = model_predict.predict(X_scale)

                # show results
                st.markdown("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: RandomForestRefressor")
                st.markdown("2. Input data: ")
                for i in input_data.columns[:-3]:
                    st.write('&nbsp;'*10 + '- ' + i + ': ' + str(input_data[i][0]))
                # st.dataframe(input_data)
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.write("Prediction price: " + str(round(yhat[0],3)))  

            elif choice3_model == 'BaggingRegressor':
                # BaggingRegressor (build again, because saved model > 800MB)
                model_bg = BaggingRegressor(n_estimators=50)
                model_bg = model_bg.fit(X_train_scale, y_train)
                # prediction
                model_predict = model_bg
                yhat = model_predict.predict(X_scale)

                # show results
                st.markdown("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: BaggingRegressor")
                st.markdown("2. Input data: ")
                for i in input_data.columns[:-3]:
                    st.write('&nbsp;'*10 + '- ' + i + ': ' + str(input_data[i][0]))
                # st.dataframe(input_data)
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.write("Prediction price: " + str(round(yhat[0],3))) 
        else:
            st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
            st.write("To use specific models to predict avocado prices, please follow these steps and get the predict results.")
            st.write("**Step 1: Choose the way to input data: input a specific features directlly or load a csv file**")
            st.image('material/P1_guidance_1.jpg')
            st.write("**Step 2: Input features information**")

            st.write('2.1. If you want to input features one by one directly:')
            st.write("Type of model, Type of avocado hass and Region:")
            st.image('material/P1_guidance_2.jpg')
            st.write("Features about volumne:")
            st.image('material/P1_guidance_3.jpg')
            st.write("Features about number of bags:")
            st.image('material/P1_guidance_4.jpg')
            st.write("2.2. If you want load a whole csv file, choose type of model and click 'Browse files' to load csv file. Download a sample of the csv file if needed.")    
            st.image('material/P1_guidance_8.jpg')

            st.write("**Step 3: Click 'Show predict results'**")
            st.image('material/P1_guidance_5.jpg')
            st.write("**Step 4: Read the forecast results.**")
            st.write("In case of input features directly:")
            st.image('material/P1_guidance_9.jpg')
            st.write("In case of loading csv file directly:")
            st.image('material/P1_guidance_10.jpg')

            st.write("**Step 5: In case of inputing features by loading csv file, if you want to download predict results, click 'Download data as CSV'**")
            st.image('material/P1_guidance_6.jpg')

            
    elif choice3_input == 'Load file':

        # sidebar
        # upload template
        upload_template = pd.read_csv('material/upload_template.csv')
        download_template = upload_template.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label="Download template as CSV",
                            data=download_template,
                            file_name='template.csv',
                            mime='text/csv',
                            )

        # upload file
        try:
            uploaded_file = st.sidebar.file_uploader('Upload data', type = ['csv'])
            # dir_file = 'material/' + uploaded_file.name
            # st.write(uploaded_file)

        except Exception as failGeneral:        
            print("Fail system, please call developer...", type(failGeneral).__name__)
            print("Description:", failGeneral)
        finally:
            print("File uploaded")



        # show results
        if st.sidebar.button("Show predict results"):
            uploaded_data = pd.read_csv(uploaded_file)
            uploaded_data = uploaded_data.drop_duplicates()
            uploaded_data = uploaded_data.loc[uploaded_data['region'] != "TotalUS",:]
            uploaded_data = uploaded_data.loc[:, ~uploaded_data.columns.str.contains('^Unnamed')]
            # X_new, y_new = model_pre.transform(uploaded_data)
            # X_scale = model_scaler.transform(X_new)
            X_new, y_new = pre_processing(uploaded_data)
            X_scale = scaling(X_new)
        # content
            if choice3_model == 'ExtraTreesRegressor':  
                # ExtraTreesRegressor (build again, because saved model > 800MB)
                model_ex = ExtraTreesRegressor(n_estimators=50, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            max_features='sqrt', 
                                            max_depth=None, 
                                            bootstrap=False)
                model_ex = model_ex.fit(X_train_scale, y_train) 

                # prediction
                model_predict = model_ex
                yhat = model_predict.predict(X_scale)
                # r2_score = r2_score(y_new, yhat)
                df_results = pd.concat([uploaded_data.reset_index(drop=True), pd.DataFrame(yhat, columns=['prediction'])], axis=1)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: ExtraTreesRegressor")
                st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(df_results)
                download_results = df_results.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results.csv',
                                    mime='text/csv',
                                    )

            elif choice3_model == 'RandomForestRefressor':
                # RandomForestRegressor (build again, because saved model > 800MB)
                model_rf = RandomForestRegressor(n_estimators=50, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            max_features='sqrt', 
                                            max_depth=None, 
                                            bootstrap=False)
                model_rf = model_rf.fit(X_train_scale, y_train)

                # prediction
                model_predict = model_rf
                yhat = model_predict.predict(X_scale)
                # r2_score = r2_score(y_new, yhat)
                df_results = pd.concat([uploaded_data.reset_index(drop=True), pd.DataFrame(yhat, columns=['prediction'])], axis=1)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: RandomForestRefressor")
                st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(df_results)
                download_results = df_results.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results.csv',
                                    mime='text/csv',
                                    )

            elif choice3_model == 'BaggingRegressor':
                # BaggingRegressor (build again, because saved model > 800MB)
                model_bg = BaggingRegressor(n_estimators=50)
                model_bg = model_bg.fit(X_train_scale, y_train)

                # prediction
                model_predict = model_bg
                yhat = model_predict.predict(X_scale)
                # r2_score = r2_score(y_new, yhat)
                df_results = pd.concat([uploaded_data.reset_index(drop=True), pd.DataFrame(yhat, columns=['prediction'])], axis=1)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information</h5>", unsafe_allow_html=True)
                st.write("1. Model name: BaggingRegressor")
                st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(df_results)
                download_results = df_results.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results.csv',
                                    mime='text/csv',
                                    )     

        else:
            st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
            st.write("To use specific models to predict avocado prices, please follow these steps and get the predict results.")
            st.write("**Step 1: Choose the way to input data: input a specific features directlly or load a csv file**")
            st.image('material/P1_guidance_1.jpg')
            st.write("**Step 2: Input features information**")

            st.write('2.1. If you want to input features one by one directly:')
            st.write("Type of model, Type of avocado hass and Region:")
            st.image('material/P1_guidance_2.jpg')
            st.write("Features about volumne:")
            st.image('material/P1_guidance_3.jpg')
            st.write("Features about number of bags:")
            st.image('material/P1_guidance_4.jpg')
            st.write("2.2. If you want load a whole csv file, choose type of model and click 'Browse files' to load csv file. Download a sample of the csv file if needed.")    
            st.image('material/P1_guidance_8.jpg')

            st.write("**Step 3: Click 'Show predict results'**")
            st.image('material/P1_guidance_5.jpg')
            st.write("**Step 4: Read the forecast results.**")
            st.write("In case of input features directly:")
            st.image('material/P1_guidance_9.jpg')
            st.write("In case of loading csv file directly:")
            st.image('material/P1_guidance_10.jpg')

            st.write("**Step 5: In case of inputing features by loading csv file, if you want to download predict results, click 'Download data as CSV'**")
            st.image('material/P1_guidance_6.jpg')
       


#####################################################################################################
# PROJECT 2 - TIME SERIES
#####################################################################################################

# 4. Predict prices by Time series
elif choice == 'Predict prices by Time series':
    

#----------------------------------------------------------------------------------------------------
    def FP_California_organic_forecast(df, type, region, n_head):
        df_California_organic = df.loc[(df['region']==region)&(df['type']==type),['Date','AveragePrice']].sort_values(by='Date')
        df_California_organic['Month'] = df_California_organic['Date'].to_numpy().astype('datetime64[M]')
        df_California_organic['Month'] = df_California_organic['Month'].astype('datetime64[ns]')
        df_California_organic_groupby = df_California_organic.groupby(['Month']).agg({'AveragePrice': np.mean})
        df_California_organic_groupby.index = pd.to_datetime(df_California_organic_groupby.index)

        df_California_organic_groupby.index.freq = 'MS'# frequent là month
        df_California_organic_groupby.index.name="DATE"
        df = df_California_organic_groupby.reset_index()  
        df.columns = ['ds','y']

        m = Prophet(yearly_seasonality=True, \
                    daily_seasonality=False, weekly_seasonality=False) 
        m.fit(df)
        future = m.make_future_dataframe(periods=n_head, freq='M') # next 5 years
        forecast = m.predict(future)

        return m, forecast

    def choosen_region_assesment(data, type, region):
        lst_region, lst_type, lst_mae_p,lst_rmse_p = [],[],[],[]
        lst_df_mean, lst_test_mean, lst_test_std = [],[],[]
        # create dataframe with Date as index
        df_region = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        df_region = df_region.loc[(df_region['region']==region)&(df_region['type']==type),['Date','AveragePrice']].sort_values(by='Date')
        df_region['Month'] = df_region['Date'].to_numpy().astype('datetime64[M]')
        df_region['Month'] = df_region['Month'].astype('datetime64[ns]')
        df_region_groupby = df_region.groupby(['Month']).agg({'AveragePrice':np.mean})
        df_region_groupby.index = pd.to_datetime(df_region_groupby.index)
        df_region_groupby.index.freq = 'MS'# frequent là month
        df_region_groupby.index.name="DATE"

        df = df_region_groupby.reset_index()
        df.columns = ['ds','y']

        # train vs test:
        n_train = int(round(len(df_region_groupby)*0.8,0))
        train = df.iloc[:n_train,:]
        test = df.iloc[n_train:,:]
        train.tail()

        # build model
        model = Prophet(yearly_seasonality=True, \
                        daily_seasonality=False,\
                        weekly_seasonality=False)
        model.fit(train)

        # 8 months in test and 9 months to predict new values
        months = pd.date_range('2017-08-01', '2018-12-01',
                                freq='MS').strftime("%Y-%m-%d").tolist()
        future = pd.DataFrame(months)
        future.columns = ['ds']
        future['ds'] = pd.to_datetime(future['ds'])

        # use the model to make a forecast
        forecast = model.predict(future)

        # calculate MAE/RMSE between expected and predicted values for december
        y_test = test['y'].values
        y_pred = forecast['yhat'].values[:8]
        mae_p = mean_absolute_error(y_test, y_pred)
        rmse_p = sqrt(mean_squared_error(y_test, y_pred))

        # lists
        lst_region.append(region)
        lst_type.append(type)
        lst_mae_p.append(mae_p)
        lst_rmse_p.append(rmse_p)

        lst_df_mean.append(df.y.mean())
        lst_test_mean.append(test.y.mean())
        lst_test_std.append(test.y.std())

        result_df = pd.DataFrame({'region':lst_region,
                                    'type':lst_type,
                                    'mae_p':lst_mae_p,
                                    'rmse_p':lst_rmse_p,
                                    'df_mean':lst_df_mean,
                                    'test_mean':lst_test_mean,
                                    'test_std':lst_test_std})
        result_df.loc[result_df['rmse_p'] < result_df['test_std'], 'eval'] = 'good'
        result_df.loc[result_df['rmse_p'] >= result_df['test_std'], 'eval'] = 'not_good'
        return result_df
#----------------------------------------------------------------------------------------------------
    
    # sidebar 
    st.sidebar.write('Choose necessary input data:')
    input_type = st.sidebar.radio("Type",['conventional','organic'])    
    input_region = st.sidebar.selectbox("Region", lst_region)
    input_month = st.sidebar.number_input("Number of months", value = 1)

    # content
    st.header('PREDICT PRICES BY TIME SERIES')

    # show results
    if st.sidebar.button("Show forecast results"):
        time_ser_data = raw_data.drop_duplicates()     
        result_df = choosen_region_assesment(time_ser_data, input_type, input_region)
        m, forecast = FP_California_organic_forecast(time_ser_data, input_type, input_region, input_month)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]

    # content
        fig = m.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), m, forecast)

        if result_df['eval'].values == 'good':            
            t1 = 'This model is **good** for predicting ' + input_type + ' hass of ' + input_region +'.'
            t2 = 'Forecast results of ' + input_region + ' for '+ str(input_month) + ' months:'
        else: 
            t1 = 'This model is **not good** for predicting ' + input_type + 'hass of ' + input_region +'.'
            t2 = 'This forecast results of ' + input_region + ' for '+ str(input_month) + ' months are **for reference only**:'
        st.write(t1)
        st.write(t2)
        st.write("<h5 style='text-align: left; color: Black;'>1. Visualize forecast trends:</h5>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.write("<h5 style='text-align: left; color: Black;'>2. Detail forecast results:</h5>", unsafe_allow_html=True)
        st.dataframe(forecast_df)
        download_results = forecast_df.to_csv().encode('utf-8')
        # download results
        st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='forecast_results.csv',
                                    mime='text/csv',
                                    )
    else: 
        st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
        st.write("To use specific models to forecast avocado prices by region, type and the number of next months, please follow these steps and get the forecast results.")
        st.write("**Step 1: Choose input data for each of data elements: Type, Region and Number of months.**")
        st.image('material/P2_3_guidance_1.jpg')
        st.write("**Step 2: Click 'Show forecast results'**")
        st.image('material/P2_3_guidance_2.jpg')
        st.write("**Step 3: Read the forecast results.**")
        st.write("- Visualize forecast trends")
        st.write("- Detail forecast results")
        st.write("*Note: read the forecast information carefully before use it.*")
        st.image('material/P2_3_guidance_4.jpg')
        st.write("**Step 4: If you want to download forecast results, click 'Download data as CSV'**")
        st.image('material/P2_3_guidance_3.jpg')        




# 5. Conclusion and Next steps
elif choice == 'Conclusion and Next steps':

    st.header('CONCLUSION AND NEXT STEPS')

    margin_df = pd.read_csv('model/Problem4_forecast_margin_region_df.csv')
    margin_df = margin_df.loc[:, ~margin_df.columns.str.contains('^Unnamed')]
    
    st.write('**There are many ways to choose what region and what type of avocado hass to expand in the future. But we suggest 1 solution that you can consider.**')
    st.write('**Some assumptions are made to support the methodology:**')
    st.write('1. Average price always cover the cost.')
    st.write('2. You are interested in margin. When determining the desired margin, the regression model will be used to determine the volume to be produced.')
    st.write('**We will choose regions and type = organic/conventional to expand base on:**')
    st.write('1. Use Facebook Prophet model for each region and each type.')
    st.write('2. Choose regions and types that we can accurately predict the average price.')
    st.write('3. Choosse regions and types of **5-year forecast results** that have large amplitudes margin.')
    st.write('*Notes: Margin = min trend - max trend (meaning the average price rises faster)*')
    st.write('**Below are forecast results, so you can use it to make decision:**')

    st.dataframe(margin_df)
    download_results = margin_df.to_csv().encode('utf-8')
    # download results
    st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='forecast_results.csv',
                                    mime='text/csv',
                                    ) 



    







