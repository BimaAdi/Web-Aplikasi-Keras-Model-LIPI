import flask
# from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from keras.models import load_model
from model.sentiment import *

app = flask.Flask(__name__)

vocab, tokenizer, max_length, model = load_variabels()
model._make_predict_function()
graph = tf.get_default_graph()


@app.route('/')
def index():
    return "test"

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    # get the request parameters
    params = flask.request.json
    if (params == None):
        params = flask.request.args
    # if parameters are found, echo the msg parameter 
    if (params != None):
        # text = 'menaker menyisir daerah sampai pelosok indonesia'
        global graph 
        with graph.as_default():
            percent, conclusion = predict_sentiment(params.get("msg"), vocab, tokenizer, max_length, model)
        # print(percent, conclusi)
        data["percent"] = str(percent)
        data["conclusi"] = conclusion
    # return a response in json format 
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)