import base64
import io
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import seaborn as sns
import sys
import datetime
import matplotlib
from bson.objectid import ObjectId
from flask import request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pandas.plotting import table
import pymongo

matplotlib.use('Agg')

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
upload_path = r'C:\DIGITAL SERVICES PORTAL ASSESSMENT\server\datasets'
MONGO_CLIENT = 'mongodb+srv://Bharathkumar:Bharathkumar@cluster0.gnigliz.mongodb.net/dqr'
MONGO_CONNECTOR = 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1'

spark = SparkSession.builder.appName('DIGIVERZMOCK') \
    .config("spark.mongodb.input.uri", MONGO_CLIENT) \
    .config("spark.mongodb.output.uri", MONGO_CLIENT) \
    .config('spark.jars.packages', MONGO_CONNECTOR) \
    .getOrCreate()

file_name = ""
sheet_name = ""
file_type = ""
metrics = {}
db = pymongo.MongoClient(MONGO_CLIENT).dqr


def navigator_dqr(endpoints):
    @endpoints.route('/home')
    def home():
        return 'dqr_home is loaded'

    @endpoints.route('/upload', methods=['POST'])
    def upload():
        global spark, file_name, sheet_name, file_type, metrics, db
        print(request.files)
        up_file = request.files['file']
        print(up_file)
        file_path = os.path.join(upload_path, up_file.filename)
        up_file.save(file_path)
        file_name = up_file.filename
        sheet_name = request.form.get('sheet')
        file_type = request.form.get('format')
        metrics = json.loads(request.form.get('metrics'))
        print(sheet_name, file_type, file_name)
        if file_type.split('/')[-1] == 'csv':
            df = spark.read.csv('./datasets/' + file_name, header=True)
        else:
            pandas_df = pd.read_excel(io='./datasets/' + file_name, sheet_name=sheet_name)
            df = spark.createDataFrame(pandas_df)
            df.printSchema()
        try:
            df.write.format('com.mongodb.spark.sql.DefaultSource').option("uri",
                                                                          MONGO_CLIENT + "." + file_name).save()
        except:
            print("df already loaded into the database")
        print("got data request")
        return jsonify({'data': 'success'})

    @endpoints.route('/process')
    def process():
        global metrics
        if file_type.split('/')[-1] == 'csv':
            df = spark.read.csv('./datasets/' + file_name, header=True, inferSchema=True)
        else:
            pandas_df = pd.read_excel(io='./datasets/' + file_name, sheet_name=sheet_name)
            df = spark.createDataFrame(pandas_df)
        coll = db.history

        # Completeness
        ncom = df.dropna().count() / df.count() * 100
        com = round(ncom, 2)

        # Uniqueness
        nuniq = df.drop_duplicates().count() / df.count() * 100
        uniq = round(nuniq, 2)

        # Dimension
        shape = (df.count(), len(df.columns))

        # Schema
        schema = df.dtypes
        print(schema)

        # Summary Statistics
        df_pd = df.toPandas()
        df_pd_describe = df_pd.describe().to_dict()
        base64_describe = df_pd_describe

        # Numerical Distribution
        fig, ax = plt.subplots(figsize=(20, 10))
        df_pd.hist(ax=ax)
        num_stringIObytes = io.BytesIO()
        fig.savefig(num_stringIObytes, format='png')
        num_stringIObytes.seek(0)
        base64_num = base64.b64encode(num_stringIObytes.read())

        # Correlation heat map
        corr_data_temp = {}
        cols = []
        for a in schema:
            if a[-1] != 'string' and a[-1] != 'timestamp':
                corr_data_temp[a[0]] = []
                cols.append(a[0])
            for b in schema:
                if a[-1] != 'string' and b[-1] != 'string' and a[-1] != 'timestamp' and b[-1] != 'timestamp':
                    corr_data_temp[a[0]].append(df.corr(a[0], b[0]))
        fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
        sns.heatmap(pd.DataFrame(corr_data_temp, index=corr_data_temp.keys()), ax=ax, annot=True, cmap='viridis',
                    linewidth=2, linecolor='black', cbar=False)
        corr_stringIObytes = io.BytesIO()
        fig.savefig(corr_stringIObytes, format='png')
        corr_stringIObytes.seek(0)
        base64_correlation = base64.b64encode(corr_stringIObytes.read())

        # Categorical Distribution
        base64_cat = {}
        for c in df.dtypes:
            if c[-1] == 'string':
                df_temp = df.groupBy(c[0]).count().orderBy(col('count').desc()).take(5)
                temp = []
                for i in df_temp:
                    temp.append(i.asDict())
                a = i.asDict()
                temp_stringIObytes = io.BytesIO()
                pd.DataFrame(temp).plot(kind='bar', x=list(a.keys())[0], figsize=(12, 5), rot=0,
                                        color='red').figure.savefig(temp_stringIObytes, format='jpg')
                temp_stringIObytes.seek(0)
                base64_cat[c[0]] = base64.b64encode(temp_stringIObytes.read())
        for i in base64_cat:
            base64_cat[i] = str(base64_cat[i])[2:-1]

        data_temp = {
            'shape': shape,
            'unique': uniq,
            'complete': com,
            'num_dist': str(base64_num)[2:-1],
            'corr': str(base64_correlation)[2:-1],
            'schema': schema,
            'cat_dist': base64_cat,
            'describe': base64_describe
        }
        data_final = {}
        for i in metrics:
            if metrics[i]:
                data_final[i] = data_temp[i]
        try:
            coll.insert_one({'fileName': file_name, 'fileType': file_type, 'sheetName': sheet_name, 'data':data_final, 'date': datetime.datetime.now()})
        except:
            print("Inserting report into history failed")
        return jsonify({'data': data_final})

    @endpoints.route('/history')
    def history():
        coll = db.history
        data = []
        for i in coll.find({}):
            i["_id"] = str(i["_id"])
            data.append(i)
        return jsonify({'data': data})

    @endpoints.route('/view/<id>')
    def view(id):
        coll = db.history
        data = coll.find_one({"_id": ObjectId(id)}, {'data': 1, '_id': 0})
        return {'data': data}
