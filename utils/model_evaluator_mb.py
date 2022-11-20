import base64
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, \
    ConfusionMatrixDisplay, balanced_accuracy_score, jaccard_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


def build_evaluate_model(model_name, model, X, y, X_train, y_train, X_test, y_test,ct,lb):
    """
      Build and Evaluate Model Performance

              1) model: object of classifier
              2) X_train: X_train
              3) y_train: y_train
              4) X_test: X_test
              5) y_test: y_test
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_train_proba = model.predict_proba(X_train)

    train_results_data = {'model_name': type(model).__name__}
    test_results_data = {'model_name': type(model).__name__}

    #   Calculating Performance Metrics
    print("\t Estimator Performance \t")
    print("\t ----------------- \t")

    print()

    print("Training Performance")
    print("--------------------")
    fpr, tpr, thresh = roc_curve(y_train, y_pred_train_proba[:, 1])

    print("Accuracy Score: ", accuracy_score(y_train, y_pred_train))
    train_results_data['Accuracy Score'] = accuracy_score(y_train, y_pred_train)
    print("Balanced Accuracy Score: ", balanced_accuracy_score(y_train, y_pred_train))
    train_results_data['Balanced Accuracy Score'] = balanced_accuracy_score(y_train, y_pred_train)

    print("Jaccard Similarity Score: ", jaccard_score(y_train, y_pred_train))
    train_results_data['Jaccard Similarity Score'] = jaccard_score(y_train, y_pred_train)

    print("Roc Score: ", roc_auc_score(y_train, y_pred_train_proba[:, 1]))
    train_results_data['Roc Score'] = roc_auc_score(y_train, y_pred_train_proba[:, 1])

    print("Classification Report: \n", classification_report(y_train, y_pred_train))
    train_results_data['Classification Report'] = classification_report(y_train, y_pred_train,output_dict=True)
    print()
    curve = RocCurveDisplay(fpr=fpr, tpr=tpr)
    fig, ax = plt.subplots()
    curve.plot(ax=ax)
    roc_stringIObytes = io.BytesIO()
    fig.savefig(roc_stringIObytes, format='png')
    roc_stringIObytes.seek(0)
    base64_roc = base64.b64encode(roc_stringIObytes.read())
    train_results_data['Roc Curve'] = 'data:image/jpeg;base64,'+str(base64_roc)[2:-1]

    matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train, y_pred_train))
    fig, ax = plt.subplots()
    matrix.plot(ax=ax)
    matrix_stringIObytes = io.BytesIO()
    fig.savefig(matrix_stringIObytes, format='png')
    matrix_stringIObytes.seek(0)
    base64_matrix = base64.b64encode(matrix_stringIObytes.read())
    train_results_data['Confusion Matrix'] = 'data:image/jpeg;base64,'+str(base64_matrix)[2:-1]

    print()

    print("Testing Performance")
    print("--------------------")
    fpr, tpr, thresh = roc_curve(y_test, y_pred_proba[:, 1])
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))
    test_results_data['Accuracy Score'] = accuracy_score(y_test, y_pred)

    print("Balanced Accuracy Score: ", balanced_accuracy_score(y_test, y_pred))
    test_results_data['Balanced Accuracy Score'] = balanced_accuracy_score(y_test, y_pred)

    print("Jaccard Similarity Score: ", jaccard_score(y_test, y_pred))
    test_results_data['Jaccard Similarity Score'] = jaccard_score(y_test, y_pred)

    print("Roc Score: ", roc_auc_score(y_test, y_pred_proba[:, 1]))
    test_results_data['Roc Score'] = roc_auc_score(y_test, y_pred_proba[:, 1])

    print("Classification Report: \n", classification_report(y_test, y_pred))
    test_results_data['Classification Report']  = classification_report(y_test, y_pred,output_dict=True)
    print()
    curve = RocCurveDisplay(fpr=fpr, tpr=tpr)
    fig, ax = plt.subplots()
    curve.plot(ax=ax)
    roc_stringIObytes = io.BytesIO()
    fig.savefig(roc_stringIObytes, format='png')
    roc_stringIObytes.seek(0)
    base64_roc = base64.b64encode(roc_stringIObytes.read())
    test_results_data['Roc Curve'] = 'data:image/jpeg;base64,'+str(base64_roc)[2:-1]

    matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    fig, ax = plt.subplots()
    matrix.plot(ax=ax)
    matrix_stringIObytes = io.BytesIO()
    fig.savefig(matrix_stringIObytes, format='png')
    matrix_stringIObytes.seek(0)
    base64_matrix = base64.b64encode(matrix_stringIObytes.read())
    test_results_data['Confusion Matrix'] = 'data:image/jpeg;base64,'+str(base64_matrix)[2:-1]
    cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=10)
    X = ct.transform(X)
    y = lb.transform(y)
    cross_results = cross_val_score(model, X, y, cv= cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross Validated Accuracy: {np.mean(cross_results) * 100:.2f}%")
    model_params = dict(itertools.islice(model.get_params().items(), 5))
    return {'training_result': train_results_data, 'testing_result': test_results_data, 'cross_validation': np.mean(cross_results),
            'model_params': model_params}
