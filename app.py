import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
graph=tf.get_default_graph()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global graph
    with graph.as_default():
        int_features = [int(x) for x in request.form.values()]
        final_features = np.array(int_features).reshape(1,-1)
        prediction = model.predict(final_features)
    prediction=(prediction>0.5)
    output = prediction 

    return render_template('index.html', prediction_text='Sensory Disorder: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)