import numpy as np
import keras

Secs = [('степень двойки', [1, 2, 4, 8, 16, 32, 64, 128, 256]), 
('увеличение на единицу', [1, 2, 3, 4, 5, 6, 7]), 
('фибонначи', [1, 2, 3, 5, 8, 13, 21]), 
('белл', [1, 2, 5, 15, 52, 203, 877])]


class LSTModel:
    def __init__(self):
        self.len_input, self.features, self.steps, self.n_seq= 4, 1, 2, 2
        self.model = keras.models.Sequential()
        self.make_model()


    def make_sec(self, list_massive):
        x_input = np.array([list_massive[i:i+self.len_input] for i in range(len(list_massive) - self.len_input)])
        x = x_input.reshape((x_input.shape[0], self.n_seq, self.steps, self.features))
        x_input = np.array(x_input[2]) # доработать
        return x_input, x


    def make_input(self, x): 
        return x.reshape((1, self.n_seq, self.steps, self.features))


    def make_output(self, list_massive):
        y = [list_massive[i] for i in range(self.len_input, len(list_massive))]
        return np.array(y)

 
    def make_model(self):
        self.model.add(keras.layers.TimeDistributed(keras.layers.convolutional.Conv1D(activation='tanh')))
        self.model.add(keras.layers.TimeDistributed(keras.layers.convolutional.MaxPooling1D(pool_size=2)))
        self.model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        self.model.add(keras.layers.LSTM(100, activation = 'tanh'))
        self.model.add(keras.layers.Dense(1))
        self.model.compile()


    def predict_number(self, x, y, x_input):
        self.model.fit(x, y, epochs=100)
        y_result = self.model.predict(x_input)
        return round(int(y_result), 2)


    def print_results(self, tuple_x, input, result):
        print("Последовательность: ", tuple_x[0], tuple_x[1])
        print("Результат для выборки ", input,  ": " , result)



lstm = LSTModel()

for i in Secs:
            input, x = lstm.make_sec(i[1])
            np_input = lstm.make_input(input)
            output = lstm.make_output(i[1])
            result = lstm.predict_number(x, output, np_input)
            lstm.print_results(i, input, result)
