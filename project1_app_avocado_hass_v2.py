# Import library
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

# import import_ipynb
# from lib.function_lib import *

data = pd.read_csv('data/avocado.csv')


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


#####################################################################################################
# PROJECT 1 - REGRESSION
#####################################################################################################

# 3. Predict prices by Regression
elif choice == 'Predict prices by Regression':

#####################################################################################################
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
        df.columns = ['Date', 'AveragePrice', 'Total_Volume', 'PLU_4046','PLU_4225','PLU_4770',
              'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 'type', 'year',
              'region']

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
        # lst_cont_fea.remove('AveragePrice')        
        X = pd.concat([df.loc[:, df.columns.str.startswith("oh_")], \
               df[lst_cont_fea]], axis=1)
        
        y = df['AveragePrice']

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

#####################################################################################################

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
        fea_region = st.sidebar.selectbox("Region",['California','region2'])
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
            lst_input = [fea_type, fea_region, fea_PLU_4046, fea_PLU_4225, fea_PLU_4770, fea_Total_Volume, fea_Small_Bags, fea_Large_Bags, fea_XLarge_Bags, fea_Total_Bags, '','','']
            col_names = ['type', 'region', 'PLU_4046', 'PLU_4225', 'PLU_4770', 'Total Volume', 'Small Bags', 'Large Bags', 'XLarge Bags', 'Total Bags', 'AveragePrice', 'Date', 'year']
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
                st.markdown('ExtraTreesRegressor')
                st.write("yhat")
                st.write(yhat)                          


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
                st.markdown('RandomForestRefressor') 
                st.write("yhat")
                st.write(yhat)  

            elif choice3_model == 'BaggingRegressor':
                # BaggingRegressor (build again, because saved model > 800MB)
                model_bg = BaggingRegressor(n_estimators=50)
                model_bg = model_bg.fit(X_train_scale, y_train)
                # prediction
                model_predict = model_bg
                yhat = model_predict.predict(X_scale)

                # show results
                st.markdown('BaggingRegressor')  
                st.write("yhat")
                st.write(yhat) 

            # elif choice3_model == 'KNeighborsRegressor':
                # # KNeighborsRegressor(build again, because saved model > 800MB)
                # model_kn = KNeighborsRegressor(n_neighbors=5, 
                #                weights='uniform', 
                #                algorithm='auto', 
                #                leaf_size=30, 
                #                p=2, 
                #                metric='minkowski', 
                #                metric_params=None, 
                #                n_jobs=None)
                # model_kn = model_kn.fit(X_train_scale, y_train)
                # # prediction
                # model_predict = model_kn
                # yhat = model_predict.predict(X_scale)

                # show results
                # st.markdown('KNeighborsRegressor')
                # st.write("yhat")
                # st.code("yhat: " + yhat)  


            # elif choice3_model == 'ANN':
                # ## ANN (from saved model)
                # model_ann = load_model('model/Problem1_ANN_model.h5')                
                # # prediction
                # model_predict = model_ann
                # yhat = model_predict.predict(X_scale)

                # show results
                # st.markdown('ANN')
                # st.write("yhat")
                # st.code("yhat: " + yhat)  


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

        # show results
        if st.sidebar.button("Show predict results"):
            uploaded_data = pd.read_csv(dir_file)
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
                st.markdown('ExtraTreesRegressor')
                st.write("yhat")
                st.dataframe(df_results)
                download_results = df_results.to_csv().encode('utf-8')
                # st.markdown('r2_score: ' + str(r2_score))

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
                st.markdown('ExtraTreesRegressor')
                st.write("yhat")
                st.dataframe(df_results)
                download_results = df_results.to_csv().encode('utf-8')
                # st.markdown('r2_score: ' + str(r2_score))

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
                st.markdown('ExtraTreesRegressor')
                st.write("yhat")
                st.dataframe(df_results)
                download_results = df_results.to_csv().encode('utf-8')
                # st.markdown('r2_score: ' + str(r2_score))

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results.csv',
                                    mime='text/csv',
                                    )

            # elif choice3_model == 'KNeighborsRegressor':
                # # KNeighborsRegressor(build again, because saved model > 800MB)
                # model_kn = KNeighborsRegressor(n_neighbors=5, 
                #                weights='uniform', 
                #                algorithm='auto', 
                #                leaf_size=30, 
                #                p=2, 
                #                metric='minkowski', 
                #                metric_params=None, 
                #                n_jobs=None)
                # model_kn = model_kn.fit(X_train_scale, y_train)
                # # prediction
                # model_predict = model_kn
                # yhat = model_predict.predict(X_scale)
                # r2_score = r2_score(y_new, yhat)
                # show results
                # st.markdown('KNeighborsRegressor')
                # st.write("yhat")
                # st.dataframe(yhat)
                # st.markdown('r2_score: ' + str(r2_score))

            # elif choice3_model == 'ANN':
                ## ANN (from saved model)
                # model_ann = load_model('model/Problem1_ANN_model.h5')
                # input_shape = X_train_scale.shape[1]
                # model_ann = keras.Sequential([
                #           layers.Dense(32, activation='relu', input_shape=[input_shape]),
                #           layers.Dense(16, activation='relu'),
                #           layers.Dense(8, activation='relu'),
                #           layers.Dense(1, activation='linear'),
                #         ])
                # early_stopping = EarlyStopping(
                #             min_delta=0.001, # minimium amount of change to count as as improvement
                #             patience=20, # how many epochs to wait before stopping
                #             restore_best_weights=True,
                #         )
                # model_ann.compile(
                #             optimizer='adam',
                #             loss=['mae','mse']
                #             # metrics=['mse','mae']
                #         )
                # history = model_ann.fit(
                #             X_train_scale, y_train,
                #             validation_data=(X_test_scale, y_test),
                #             batch_size=32, # default = 32
                #             epochs=200,
                #             callbacks=[early_stopping], # put your callbacks in a list
                #             verbose=1, # turn off training log
                #         )
                # prediction
                # model_predict = model_ann
                # yhat = model_predict.predict(X_scale)
                # r2_score = r2_score(y_new, yhat)
                # show results
                # st.markdown('ANN')
                # st.write("yhat")
                # st.dataframe(yhat)
                # st.markdown('r2_score: ' + str(r2_score))


       


#####################################################################################################
# PROJECT 2 - TIME SERIES
#####################################################################################################

# 4. Predict prices by Time series
elif choice == 'Predict prices by Time series':

    def FP_California_organic_forecast(df, region, n_head):
        df_California_organic = df.loc[(df['region']==region)&(df['type']=='organic'),['Date','AveragePrice']].sort_values(by='Date')
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

     
    st.header('PREDICT PRICES BY TIME SERIES')

    input_month = st.sidebar.number_input("Number of months", value = 1)
    input_region = st.sidebar.selectbox("Region",['California','Albany','Atlanta','region2'])
    # show results
    if st.sidebar.button("Show predict results"):
        uploaded_data = pd.read_csv('data/avocado.csv')
        uploaded_data = uploaded_data.drop_duplicates()
        uploaded_data = uploaded_data.loc[uploaded_data['region'] != "TotalUS",:]
        uploaded_data = uploaded_data.loc[:, ~uploaded_data.columns.str.contains('^Unnamed')]
        m, forecast = FP_California_organic_forecast(uploaded_data, input_region, input_month)
    # content
        fig = m.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), m, forecast)
        t = 'Forcast results of ' + input_region + ' for '+ str(input_month) + ' months'  
        st.write(t)
        st.pyplot(fig)



# 5. Conclusion and Next steps
elif choice == 'Conclusion and Next steps':
    st.header('CONCLUSION AND NEXT STEPS')
