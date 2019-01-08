from keras import Input, Model, metrics
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, CuDNNLSTM, Bidirectional, \
    CuDNNGRU, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import os
from keras.utils import to_categorical


class CategoryClassifier:
    def __init__(self):
        pass

    def training(self, input_data, output_data, val_input, val_output):

        print("Get category number")
        self._get_category_num(output_data)

        print("Build tokenizer")
        self._build_tokenizer(input_data)

        print("Convert data")
        input_data, output_data = self._convert_data(input_data, output_data)
        print(input_data)
        print(output_data)

        print("Build network")
        self._build_network(input_data, output_data)

        print("Training")
        self._training(input_data, output_data, val_input, val_output)

    def validation(self, data):
        pass

    def prediction(self, data):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _build_tokenizer(self, input_data):
        self._tokenizer = Tokenizer(char_level=True)
        self._tokenizer.fit_on_texts(input_data)

    def _build_network(self, input_data, output_data):
        print(input_data[0])
        print(len(input_data[0]))
        print(output_data[0])
        print(len(output_data[0]))
        emb_size = 256
        num_tokens = len(self._tokenizer.word_index) + 1
        _input = Input(shape=(len(input_data[0]),))
        x = Embedding(num_tokens, emb_size, trainable=True)(_input)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(128, return_sequences=True))(x)
        x = Conv1D(256, 3, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(pool_size=3)(x)
        x = Flatten()(x)
        output = Dense(len(output_data[0]), activation='softmax', name="output_dense")(x)

        self._model = Model(inputs=_input, outputs=[output])

        self._model.summary()

    def _get_category_num(self, output_data):
        self._num_of_category = len(set(output_data))

    def _training(self, input_data, output_data, val_input, val_output):
        model_save_path = './model/'
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        model_path = model_save_path + '{epoch:02d}-{val_loss:.4f}.hdf5'

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                        verbose=1, save_best_only=True)

        self._model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[metrics.categorical_accuracy])
        self._model.fit(input_data,
                        output_data,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=256,
                        callbacks=[cb_checkpoint])

    def _convert_data(self, input_data, output_data):
        input_data = self._tokenizer.texts_to_sequences(input_data)
        maxlen = 0
        for data in input_data:
            if len(data) > maxlen:
                maxlen = len(data)
        return pad_sequences(input_data, maxlen=maxlen), to_categorical(output_data)
