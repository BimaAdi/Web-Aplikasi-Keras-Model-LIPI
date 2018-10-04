import string
import re
import os
import keras
import tensorflow as tf
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import model_from_json
from sklearn.externals import joblib

def load_variabels():
    # global model,graph
    vocab = joblib.load('model/evaluatePartVocab.joblib')
    tokenizer = joblib.load('model/evaluatePartTokenizer.joblib')
    max_length = joblib.load('model/evaluatePartMax_length.joblib')
    model = load_model('model/model.h5')
    # graph = tf.get_default_graph()
    return vocab, tokenizer, max_length, model

def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
	# integer encode
	encoded = tokenizer.texts_to_sequences(docs)
        #print(encoded)
	# pad sequences
	padded = pad_sequences(encoded, maxlen=max_length, padding='post')
	#print(padded)
	return padded

def predict_sentiment(review, vocab, tokenizer, max_length, model):
	# clean review
	line = clean_doc(review, vocab)
	print('ini line',line)
	# encode and pad review
	padded = encode_docs(tokenizer, max_length, [line])
    #print('padded',padded)
	# predict sentiment
	yhat = model.predict(padded, verbose=0)
	print('yhat',yhat)
	# retrieve predicted percentage and label
	percent_pos = yhat[0,0]
	print(round(percent_pos))
	if round(percent_pos) == 0:
		return (1-percent_pos), 'NEGATIVE'
	return percent_pos, 'POSITIVE'
