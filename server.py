##REF : https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
loaded_model = pickle.load(open('clf_model.pkl','rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle','rb'))

data_predict = {'SQL_STATEMENT_TO_PREDICT': ["kalau ada sumur di ladang"]}
new_data = pd.DataFrame(data_predict, columns = ['SQL_STATEMENT_TO_PREDICT'])

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict2', methods=['POST'])
def predict2():
    data_sql_statement = request.form['sql_statement']
    #new_data = pd.DataFrame(data_sql_statement, columns=['SQL_STATEMENT_TO_PREDICT'])
    #pred = loaded_model.predict(loaded_vectorizer.transform(new_data['SQL_STATEMENT_TO_PREDICT']))
    arr = np.array([data_sql_statement])
    pred = loaded_model.predict(loaded_vectorizer.transform(arr.ravel()))

    predicted_output2 = (predicted_output.to_string(index=False))
    #return render_template("result.html", data=predicted_output2)

    predicted_output = pd.DataFrame(pred, columns=['Result'])
    return render_template("result.html",
                           tables=[predicted_output.to_html(classes='data', index=False)],
                           titles=predicted_output.columns.values)
    
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    # prediction = model.predict([[np.array(data['exp'])]])
    #prediction = loaded_model.predict(loaded_vectorizer.transform(new_data['SQL_STATEMENT_TO_PREDICT']))
    prediction = loaded_model.predict(loaded_vectorizer.transform([[np.array(data['sql_statement'])]]))

    # Take the first value of prediction
    #predicted_output = pd.DataFrame({'SQL_STATEMENT_TO_PREDICT': new_data['SQL_STATEMENT_TO_PREDICT'],
    #                                 'PREDICTED_BYTES_STREAMED_CLUSTER': prediction})
    output = prediction[0]
    print(">>> OUTPUT >>> ", output)
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
