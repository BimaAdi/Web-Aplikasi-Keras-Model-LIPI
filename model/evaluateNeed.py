import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.externals import joblib


# load text-------------------------------------------------
text = 'menaker menyisir daerah sampai pelosok indonesia'

# load the vocabulary -------------------------------------
vocab = joblib.load('evaluatePartVocab.joblib')

#load tokenizer-------------------------------------------
tokenizer = joblib.load('evaluatePartTokenizer.joblib')

# load max_length-----------------------------------------------
max_length = joblib.load('evaluatePartMax_length.joblib')

# load model------------------------------------------------------
model = load_model('model.h5')

# classify a review as negative or positive
# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
	# integer encode
	encoded = tokenizer.texts_to_sequences(docs)
        #print(encoded)
	# pad sequences
	padded = pad_sequences(encoded, maxlen=max_length, padding='post')
	#print(padded)
	return padded

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

# # clean review
# line = clean_doc(text, vocab)
# print('ini line',line)
# # encode and pad review
# padded = encode_docs(tokenizer, max_length, [line])
# #print('padded',padded)
# # predict sentiment
# yhat = model.predict(padded, verbose=0)
# print('yhat',yhat)
# # retrieve predicted percentage and label
# percent_pos = yhat[0,0]
# print(round(percent_pos))
# if round(percent_pos) == 0:
# 	print(1-percent_pos, 'NEGATIVE')
# print(percent_pos, 'POSITIVE')


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

percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))