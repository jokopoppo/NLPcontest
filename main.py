# import deepcut
import LSTMmodel
import json
from gensim.models import KeyedVectors

data_train = json.load(open('data.json', 'r', encoding='utf-8'))
word_vec_file = KeyedVectors.load_word2vec_format('E:\\CPE#Y4\\NLP\\wiki.th.vec')
print("Init Data")
# LSTMmodel.LSTM_train(data_train, word_vec_file)
LSTMmodel.predict_data(data_train, word_vec_file)
