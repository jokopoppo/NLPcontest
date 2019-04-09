# import deepcut
from LSTMmodel import LSTM_train
import json
import fasttext
from gensim.models import KeyedVectors

# fasttext.skipgram('token_input.json', 'fasttext_sg_model',silent=0, dim=100)
data_train = json.load(open('data.json', 'r', encoding='utf-8'))
word_vec_file = KeyedVectors.load_word2vec_format('E:\\CPE#Y4\\NLP\\wiki.th.vec')
print("Init Data")
LSTM_train(data_train, word_vec_file)

# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(data_train, f, ensure_ascii=False)
