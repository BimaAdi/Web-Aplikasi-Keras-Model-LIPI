import os
from werkzeug import secure_filename
import flask
from flask import render_template, request, redirect, url_for
from model.sentiment import *
from seed.all_prediction_result import *

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_file/'

# ambil fungsi dari model\sentiment.py
# menggunakan graph agar model bisa berjalan di route
vocab, tokenizer, max_length, model = load_variabels()
model._make_predict_function()
graph = tf.get_default_graph()


@app.route('/')
def index():
    return render_template('check.html', data=prediction_result)
    #return flask.jsonify(prediction_result)


# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {}
    # get the request parameters
    #params = flask.request.json
    params = request.form
    if (params == None):
        params = flask.request.args
    if (params != None):
        # pastikan tidak kosong
        # if(params.get("text") != ""):

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
    #return flask.jsonify(data)
    return redirect(url_for('index'))

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(filename)

        input_file = open(filename, 'r')
        for line in input_file:
            # print(line)
            # prediksi setiap baris
            global graph 
            with graph.as_default():
                percent, conclusion = predict_sentiment(line, vocab, tokenizer, max_length, model)

             # Input data hasil predict_sentiment model ke list 
            data = {}
            data["text"] = line
            data["percent"] = str(percent)
            data["conclusi"] = conclusion

            # print(data)
            #Input data ke prediction result
            prediction_result.append(data)
                

        input_file.close()

        os.remove(filename)
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)