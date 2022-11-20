from flask import Flask, request
import base64
import io
import matplotlib.pyplot as plt
import os, json
import pandas as pd
import seaborn as sns
import sys, datetime
from pymongo import MongoClient

from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from bson import json_util, ObjectId
from pyspark.sql import SparkSession
import itertools
# from sktime.forecasting.sarimax import SARIMAX
# from sktime.forecasting.trend import TrendForecaster
# from autots import AutoTS
import joblib
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, DoubleType, LongType
# from sktime.performance_metrics.forecasting import     MeanAbsolutePercentageError
# from prophet.diagnostics import performance_metrics,cross_validation
import pycaret.time_series as pycaret_timeseries


import numpy as np
# Loading Models
model_prophet = 'sf_models/prophet_engine2.model'
model_expo = 'sf_models/expo_smooth'


# MONGO INSTANCE
MONGO_CLIENT = 'mongodb+srv://Bharathkumar:Bharathkumar@cluster0.gnigliz.mongodb.net'
db = MongoClient(MONGO_CLIENT)['sf']
coll = db.housing_sales
coll2 = db.history
# Global Var
spark = SparkSession.builder \
    .appName("app_sf") \
    .getOrCreate()

df = ""
pandas_df = ""
monthly_frame = ""
model = ""
months = 0

bi_path_prophet = r'C:\DIGITAL SERVICES PORTAL ASSESSMENT\prophet.csv'
bi_path_expo = r'C:\DIGITAL SERVICES PORTAL ASSESSMENT\exposmooth.csv'


def navigator_sf(endpoints):
    @endpoints.route('/home')
    def home():
        return 'sf_home is loaded'

    @endpoints.route('/upload', methods=['POST'])
    def upload():
        global df, model, months, pandas_df
        model = request.form.get('model')
        months = int(request.form.get('months'))
        data = []
        for i in coll.find({}, {'_id': 0}):
            i['datesold'] = datetime.datetime.strptime(str(i['datesold']).split()[0], '%Y-%m-%d')
            data.append(i)
        pandas_df = pd.DataFrame(data)
        mySchema = StructType([StructField("datesold", DateType(), True)
                                  , StructField("postcode", LongType(), True)
                                  , StructField("price", LongType(), True)
                                  , StructField("propertyType", StringType(), True)
                                  , StructField("bedrooms", IntegerType(), True)
                               ])
        df = spark.createDataFrame(pandas_df, schema=mySchema)
        # df.select('datesold').show()
        return f'months and model are uploaded,{model} {months}'

    @endpoints.route('/report')
    def report():
        global df, months, coll2, monthly_frame, pandas_df
        print("Running Model training.....")
        # EDA
        schema = df.dtypes
        shape = (df.count(), len(df.columns))
        describe = df.describe().toPandas().to_dict()
        com = df.dropna('any').count() / df.count() * 100
        uniq = df.drop_duplicates().count() / df.count() * 100
        df.toPandas().hist(figsize=(12, 8))
        num_stringIObytes = io.BytesIO()
        plt.savefig(num_stringIObytes, format='png')
        num_stringIObytes.seek(0)
        base64_num = 'data:image/png;base64,' + str(base64.b64encode(num_stringIObytes.read()))[2:-1]
        monthly_frame_temp = df.select(['datesold', 'price']).toPandas().set_index('datesold')
        monthly_frame_temp.index = pd.to_datetime(monthly_frame_temp.index)
        monthly_frame = monthly_frame_temp.resample('M').sum()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(monthly_frame.index, monthly_frame.price)
        sales_stringIObytes = io.BytesIO()
        fig.savefig(sales_stringIObytes, format='png')
        sales_stringIObytes.seek(0)
        base64_sales = 'data:image/png;base64,' + str(base64.b64encode(sales_stringIObytes.read()))[2:-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='propertyType', data=pandas_df, ax=ax);
        cat_stringIObytes = io.BytesIO()
        fig.savefig(cat_stringIObytes, format='png')
        cat_stringIObytes.seek(0)
        base64_cat = 'data:image/png;base64,' + str(base64.b64encode(cat_stringIObytes.read()))[2:-1]

        # Running predictions
        data_model_stats = {}
        if model == 'prophet':
                m = joblib.load(model_prophet)
                future = m.make_future_dataframe(periods=months, freq='M', include_history=False)
                forecast = m.predict(future)
                forecasted = forecast[['ds', 'yhat']].tail(months)
                forecast.to_csv(bi_path_prophet, mode='w', header=True, index=False)
                forecasted['ds'] = forecast['ds'].astype(str)
                forecasted.set_index('ds', inplace=True)
                forecasted = forecasted.to_dict()['yhat']
                fig = m.plot(forecast)
                prophet_stringIObytes = io.BytesIO()
                fig.savefig(prophet_stringIObytes, format='png')
                prophet_stringIObytes.seek(0)
                base64_prophet = 'data:image/png;base64,' + str(base64.b64encode(prophet_stringIObytes.read()))[2:-1]
                data_model_stats['model_name'] = 'Prophet Engine Version 1'
                data_model_stats['forecasted_sales'] = forecasted
                data_model_stats['forecasted_plot'] = base64_prophet

        # if model == 'expo':
        #     final = pd.Series(monthly_frame.price, index=monthly_frame.index)
        #     s = pycaret_timeseries.setup(final, session_id=123)
        #     forecaster = pycaret_timeseries.load_model(model_expo)
        #     final_forecast = pycaret_timeseries.predict_model(forecaster, fh=months + 1)[1:]
        #     fig, ax = plt.subplots(figsize=(16, 8))
        #     sns.lineplot(x=final.index, y=final.values, ax=ax, label='actual')
        #     sns.lineplot(x=final_forecast.index, y=final_forecast.y_pred, ax=ax, label='forecast')
        #     expo_stringIObytes = io.BytesIO()
        #     fig.savefig(expo_stringIObytes, format='png')
        #     expo_stringIObytes.seek(0)
        #     base64_expo = 'data:image/png;base64,' + str(base64.b64encode(expo_stringIObytes.read()))[2:-1]
        #     temp_forecast = [0 for _ in range(len(final.index))]
        #     temp_forecast.extend(list(final_forecast['y_pred']))
        #     print(len(temp_forecast))
        #     final_forecast.index = pd.PeriodIndex(final_forecast.index.astype(str), freq='M').strftime('%Y-%m-%d')
        #     indexes = list(final.index)
        #     indexes.extend(list(final_forecast.index))
        #     temp_df = {'Date': indexes, 'Acutal': final.values, 'Forecast': temp_forecast}
        #     pd.DataFrame.from_dict(temp_df, orient='index').transpose() \
        #         .to_csv(bi_path_expo, mode='w', header=True, index=False)
        #
        #     final_forecast.index = final_forecast.index.astype(str)
        #     forecasted = final_forecast.to_dict()['y_pred']
        #
        #     data_model_stats['model_name'] = 'Exponential Smoothing'
        #     data_model_stats['forecasted_sales'] = forecasted
        #     data_model_stats['forecasted_plot'] = base64_expo

        history = []
        for i in coll2.find({}):
            i['_id'] = str(i['_id'])
            history.append(i)
        final_data = {
            'eda': {
                'shape': shape,
                'schema': schema,
                'describe': describe,
                'completeness': com,
                'uniqueness': uniq,
                'numerical distribution': base64_num,
                'monthly sales': base64_sales,
                'categorical Distribution': base64_cat,
            },
            'running_results': data_model_stats,
            'history': history

        }
        coll2.insert_one(json.loads(json.dumps(
            {'eda': final_data['eda'], 'running_results': final_data['running_results'],
             'date': str(datetime.datetime.now())})))
        return final_data

    @endpoints.route('/view/<id>')
    def view(id):
        global coll2
        data = coll2.find_one({"_id": ObjectId(id)})
        data["_id"] = str(data["_id"])
        print(data)
        return data
