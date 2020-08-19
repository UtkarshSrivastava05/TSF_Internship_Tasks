# Deployment of the ML model via Flask

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    try:
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict(final_features)
        
    except:
        return render_template('index.html', prediction_text="Hours value cannot be an \
                               alphabet or a special character! Please enter a value in \
                                   the range of 0 to 24.")
                                   
    else:
        output = round(prediction[0], 2)
    
    
        
        if float_features[0] <= 0 or float_features[0] > 24:
            return render_template('index.html', prediction_text="Enter a value in the \
                                   range of 0 to 24.")
    
        elif output > 100:
            return render_template('index.html', prediction_text="If the student studies \
                               for {} hours in a day, the student score should be \
                                   100 %".format(str(float_features[0])))
    
        else:
            return render_template('index.html', prediction_text="If the student studies \
                                   for {} hours in a day, the student score should be \
                                   {} %".format(str(float_features[0]), output))                                 
        
                                   

@app.route('/predict_api',methods=['POST'])

def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    