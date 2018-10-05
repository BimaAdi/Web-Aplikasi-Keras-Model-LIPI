import flask
from model.sentiment import *
from seed.all_prediction_result import *

app = flask.Flask(__name__)

# ambil fungsi dari model\sentiment.py
# menggunakan graph agar model bisa berjalan di route
vocab, tokenizer, max_length, model = load_variabels()
model._make_predict_function()
graph = tf.get_default_graph()


@app.route('/')
def index():
    return flask.jsonify(prediction_result)

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {}
    # get the request parameters
    params = flask.request.json
    if (params == None):
        params = flask.request.args
    if (params != None):
        # prediksi menggunakan model
        global graph 
        with graph.as_default():
            percent, conclusion = predict_sentiment(params.get("text"), vocab, tokenizer, max_length, model)

        # Input data hasil predict_sentiment model ke list 
        data["text"] = params.get("text")
        data["percent"] = str(percent)
        data["conclusi"] = conclusion

        #Input data ke prediction result
        prediction_result.append(data)

    # return a response in json format 
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)