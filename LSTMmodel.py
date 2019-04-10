def LSTM_train(data, word_vec, ):
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, Dense, GRU, Dropout, SimpleRNN, LSTM
    from keras.utils import to_categorical
    import matplotlib.pyplot as plt
    from random import shuffle
    from sklearn.metrics import confusion_matrix

    # ------------------------- Read data ------------------------------
    shuffle(data)
    for d in data:
        print(d)

    labels = [int(d[0]) for d in data]
    words = [d[1] for d in data]


    # ------------------- Extract word vectors -------------------------
    max_length = 60
    zero_vec = np.zeros(300)
    word_vectors = np.zeros((len(words), max_length, 300))

    for s in range(words.__len__()):
        for w in range(words[s].__len__()):
            try:
                word_vectors[s, w+(max_length-words[s].__len__()), :] = word_vec.wv[words[s][w]]
            except KeyError:
                word_vectors[s, w+(max_length-words[s].__len__()), :] = zero_vec
    print(word_vectors.shape)
    print(word_vectors[0])
    # --------------- Create recurrent neural network-----------------
    inputLayer = Input(shape=(60, 300))
    lstm = LSTM(10, activation='relu')(inputLayer)
    lstm = Dropout(0.5)(lstm)
    outputLayer = Dense(3, activation='softmax')(lstm)
    model = Model(inputs=inputLayer, outputs=outputLayer)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # ----------------------- Train neural network-----------------------
    model.load_weights('contest_model.h5')
    history = model.fit(word_vectors, to_categorical(labels), epochs=5, batch_size=50, validation_split=0.2)
    model.save('contest_model.h5')
    # -------------------------- Evaluation-----------------------------
    length = int(word_vectors.__len__()*20/100)
    y_pred = model.predict(word_vectors[length:, :, :])

    cm = confusion_matrix(labels[length:], y_pred.argmax(axis=1))
    print('Confusion Matrix')
    print(cm)

    plt.plot(history.history['acc'], label = 'acc')
    plt.plot(history.history['val_acc'], label = 'val_acc')
    plt.legend(loc='upper right')
    plt.show()
