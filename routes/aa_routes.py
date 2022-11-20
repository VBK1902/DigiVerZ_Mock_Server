import os
import pandas as pd
import pycaret.classification as classifier
import pycaret.regression as regressor
from flask import request

upload_path = r'C:\DIGITAL SERVICES PORTAL ASSESSMENT\server\datasets'
file_name = ""
df = ""
target = ""
domain = ""


def navigator_aa(endpoints):
    @endpoints.route('/home')
    def home():
        return {'message': 'success'}

    @endpoints.route('/upload_aa', methods=['POST'])
    def upload_df():
        try:
            global file_name, df
            up_file = request.files['file']
            file_path = os.path.join(upload_path, up_file.filename)
            up_file.save(file_path)
            file_name = up_file.filename
            df = pd.read_csv('./datasets/' + file_name)
            return {'message': 'uploaded_dataset', 'cols': list(df.columns)}
        except:
            return {'message': 'Some error in uploading dataset'}

    @endpoints.route('/upload_metrics', methods=['POST'])
    def upload_metrics():
        try:
            global target, domain
            target = request.form.get('target')
            domain = request.form.get('domain')
            return {'message': 'Successfully uploaded dataset and metrics'}
        except:
            return {'message': 'Some Error in uploading metrics and dataset!!!'}

    @endpoints.route('/report')
    def report():
        global df, target, domain
        if domain == 'Classification':
            s = classifier.setup(df, target=target)
            best = classifier.compare_models()
            results = classifier.pull()
        else:
            s = regressor.setup(df, target=target)
            best = regressor.compare_models()
            results = regressor.pull()
        return {'message': 'model training success', 'metrics': results.to_dict('records'), 'best model': str(best)}