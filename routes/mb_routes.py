import joblib
import joblib as jb
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from flask import request, jsonify
from pymongo import MongoClient
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV, \
    cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, \
    ConfusionMatrixDisplay, balanced_accuracy_score, jaccard_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from datetime import datetime
from bson import json_util, ObjectId
from utils.model_evaluator_mb import build_evaluate_model

# MONGO INSTANCE
MONGO_CLIENT = 'mongodb+srv://Bharathkumar:Bharathkumar@cluster0.gnigliz.mongodb.net'
db = MongoClient(MONGO_CLIENT)['mb']
coll = db.history

# DATASET PATH
df_path = r'C:\DIGITAL SERVICES PORTAL ASSESSMENT\server\datasets\telcomchurn.csv'
bi_path = r'C:\DIGITAL SERVICES PORTAL ASSESSMENT\performance_history.csv'

# MODELS PATH
MODEL_PATH_LR = 'mb_models/logisticRegression.model'
MODEL_PATH_ADA = 'mb_models/adaBoostClassifier.model'
MODEL_PATH_GB = 'mb_models/gradientBoostingClassifier.model'
MODEL_PATH_XGB = 'mb_models/xgb.model'
MODEL_PATH_DT = 'mb_models/decisionTreeClassifier.model'
MODEL_PATH_RF = 'mb_models/randomForest.model'

# ENCODER AND TRANSFORMER'S PATH

TRANSFORMER_PATH_COLUMN = 'mb_transformer_encoders/transform'

# Loading Models and Utility Transformers
transformer = joblib.load(TRANSFORMER_PATH_COLUMN)
lr = joblib.load(open(MODEL_PATH_LR, 'rb'))
ada = joblib.load(MODEL_PATH_ADA)
gb = joblib.load(MODEL_PATH_GB)
xgb = joblib.load(MODEL_PATH_XGB)
dt = joblib.load(MODEL_PATH_DT)
rf = joblib.load(MODEL_PATH_RF)

# Global Variables

df = ""
model = ""
target = ""
prev_results = ""


def navigator_mb(endpoints):

    @endpoints.route('/home')
    def home():
        return 'mb_home is loaded'

    @endpoints.route('/upload', methods=['POST'])
    def upload():
        global df, model, target
        model = request.form.get('model')
        target = request.form.get('target')
        df = pd.read_csv(df_path)
        print(model, target)
        print(df.shape)
        return f'metrics and model are uploaded {df.shape}'

    @endpoints.route('/train')
    def train():

        global lr, transformer, gb, ada, xgb, coll, prev_resuts, df, dt, rf, df_2
        df = pd.read_csv(df_path)
        df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
        print(df.info())
        df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
        df.fillna(df["TotalCharges"].mean())
        df_2 = df.copy()

        num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
        cat_cols_ohe = ['PaymentMethod', 'Contract', 'InternetService']  # those that need one-hot encoding
        cat_cols_le = list(set(df_2.columns[0:19]) - set(num_cols) - set(cat_cols_ohe))  # those that need label encoding

        model_dict = {'lr': lr, 'gb': gb, 'ada': ada, 'xgb': xgb, 'dt': dt, 'rf': rf}
        bi_labels = ['_id', 'model_name', 'date', 'training_accuracy', 'testing_accuracy', 'cross_validation',
                     'training_roc_score', 'testing_roc_score']

        cat_cols_le.extend(cat_cols_ohe)
        df_cols = list(set(cat_cols_le))

        final_cols = []
        for i, j in enumerate(df_2.columns):
            if j in df_cols:
                final_cols.append(i)

        ct = ColumnTransformer([('scaler', StandardScaler(), [4, 17, 18]), ('one', OneHotEncoder(), final_cols)],
                               remainder='passthrough')

        X = df_2.drop(columns=['Churn'])
        y = df_2['Churn'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

        X_train = ct.fit_transform(X_train)
        X_test = ct.transform(X_test)

        lb = LabelEncoder()
        y_train = lb.fit_transform(y_train)
        y_test = lb.transform(y_test)

        model_instance = model_dict[model]
        model_name = model
        performance_results = build_evaluate_model(model_name, model_instance, X, y, X_train, y_train, X_test, y_test,ct,lb)
        performance_results['date'] = str(datetime.now())
        performance_results['model_name'] = performance_results['training_result']['model_name']
        performance_results['Status'] = 'Active'
        prev_resuts = performance_results
        history = []
        for i in coll.find({}):
            i['_id'] = str(i['_id'])
            history.append(i)
        performance_results['Status'] = 'Processed'
        active_result = coll.insert_one(json.loads(json_util.dumps(performance_results)))
        history_bi = []
        for i in coll.find({}):
            dic = dict()
            dic['_id'] = str(i['_id'])
            dic['model_name'] = i['model_name']
            dic['date'] = i['date']
            dic['training_accuracy'] = i['training_result']['Accuracy Score']
            dic['testing_accuracy'] = i['testing_result']['Accuracy Score']
            dic['cross_validation'] = i['cross_validation']
            dic['training_roc_score'] = i['training_result']['Roc Score']
            dic['testing_roc_score'] = i['testing_result']['Roc Score']
            history_bi.append(dic)
        pd.DataFrame(data=history_bi).to_csv(bi_path, mode='w', header=True, index=False)
        performance_results['Status'] = 'Active'
        if model == 'xgb':
            print(type(performance_results))
        return {'results': {'performance': performance_results, 'history': history}}

    @endpoints.route('/predict', methods=['POST'])
    def predict():
        global model_pred, final_model
        model_pred = request.form.get('modelp')
        print(model_pred)
        model_dict = {'lr': lr, 'gb': gb, 'ada': ada, 'dt': dt, 'rf': rf}
        final_model = model_dict[model_pred]
        values = [request.form.get('gender'), request.form.get('SeniorCitizen'), request.form.get('Partner'),
                  request.form.get('Dependents'), int(request.form.get('tenure')), request.form.get('PhoneService'),
                  request.form.get('MultipleLines'), request.form.get('InternetService'),
                  request.form.get('OnlineSecurity'), request.form.get('OnlineBackup'),
                  request.form.get('DeviceProtection'), request.form.get('TechSupport'),
                  request.form.get('StreamingTV'), request.form.get('StreamingMovies'), request.form.get('Contract'),
                  request.form.get('PaperlessBilling'), request.form.get('PaymentMethod'),
                  float(request.form.get('MonthlyCharges')), float(request.form.get('TotalCharges'))]

        print(values)
        data = transformer.transform([values])


        result = final_model.predict(data)
        result_prob = final_model.predict_proba(data)
        print(result_prob)
        if result[0] == 0:
            churn = 'Yes'
            result_prob = round(result_prob[0][0] * 100,2)
        else:
            churn = 'No'
            result_prob = round(result_prob[0][1] * 100,2)
        return {'churn': churn, 'prob': result_prob}

    @endpoints.route('/view/<id>')
    def view(id):
        global coll
        data = coll.find_one({"_id": ObjectId(id)})
        data["_id"] = str(data["_id"])
        print(data)
        return data

