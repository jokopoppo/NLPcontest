# import deepcut
import LSTMmodel
import json
from gensim.models import KeyedVectors

word_vec_file = KeyedVectors.load_word2vec_format('E:\\CPE#Y4\\NLP\\wiki.th.vec')
data_train = json.load(open('data.json', 'r', encoding='utf-8'))
print("Init Data")
# LSTMmodel.LSTM_train(data_train, word_vec_file)

input_data = open('input.txt','r',encoding='utf-8-sig')

# LSTMmodel.predict_data(input_data, word_vec_file, "contest_model_100.h5", 100)
LSTMmodel.predict_data(input_data, word_vec_file, "contest_model.h5")

