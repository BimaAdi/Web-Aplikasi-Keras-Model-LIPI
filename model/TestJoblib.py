# jalankan menggunakan aplikasi-env
# coding: utf-8

# In[3]:


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

text = 'menaker menyisir daerah sampai pelosok indonesia'

# load the vocabulary -------------------------------------
vocab = joblib.load('evaluatePartVocab.joblib')

#load tokenizer-------------------------------------------
tokenizer = joblib.load('evaluatePartTokenizer.joblib')

# load max_length-----------------------------------------------
max_length = joblib.load('evaluatePartMax_length.joblib')

# load model------------------------------------------------------
model = load_model('model.h5')
# model.summary()
# print(model.inputs)
# json_file = open('architecture.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)


## -----------------------Check Version-----------------------
#print(keras.__version__) #2.2.3 -> 2.2.0
#print(tf.__version__) #1.11.0 -> 1.9.0

# In[8]:


# turn a doc into clean tokens
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


# In[10]:


# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
	# integer encode
	encoded = tokenizer.texts_to_sequences(docs)
        #print(encoded)
	# pad sequences
	padded = pad_sequences(encoded, maxlen=max_length, padding='post')
	#print(padded)
	return padded


# In[4]:


# classify a review as negative or positive
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


# In[11]:


percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))


