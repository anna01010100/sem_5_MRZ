import numpy as np
import math


class Hopfield:
    def __init__(self):
        self.loading_files()
        self.weights = self.create_weight_matrix(self.models)
        print('\nМодель: ')
        list(map(Hopfield.print_model, self.models))
        print('------------------------------------------------------------------')
        print('\nИскаженные образы: ')
        for dis in self.distorted:
            self.make_accotiation(dis)

    def loading_files(self):
        #model_name = input('Введите название файла модели')
        #distorted_name = input('Введите название файла с искаженным образом')
        model_name = 'model.txt'
        distorted_name = 'pattern.txt'
        self.models = Hopfield.file_open(model_name)
        self.distorted = Hopfield.file_open(distorted_name)
    

    @staticmethod
    def file_open(model_name):
        with open(model_name) as model_file:
            str_models = model_file.read()
            models = []
            line = []
            for i in range(len(str_models)):
                if str_models[i] == '1':
                    line.append(1)
                if str_models[i] == '0':
                    line.append(-1)
                if str_models[i] == '\n':
                    if str_models[i+1] == '\n':
                        models.append(line)
                        line = []
            models.append(line)           
        return models


    @staticmethod
    def print_model(model):
        size = int(len(model)**0.5)
        print('')
        for i in range(size):
            for j in range(size):
                if model[i*size + j] == 1:
                    print('*', end = '')
                else:
                    print('.', end = '')
            print('')
       

    @staticmethod
    def create_weight_matrix(models):
        size_weights = len(models[0])
        weights = np.zeros((size_weights, size_weights), dtype='int32')
        for model in models:
            weights += np.array([model]).T @ np.array([model])
        weights[np.diag_indices(size_weights)] = 0
        return weights


    def make_accotiation(self, dis):
        dis_np = np.array([dis])
        prev = dis_np.T
        for i in range(1000):
            if i >= 999:
                raise('Ошибка. Модель не распознана')
            next = np.sign(self.weights @ prev)
            next = Hopfield.norm(next)
            print('Итерация: ', i)
            Hopfield.print_model(next.T.flatten().tolist())
            if (next == prev).all():
                print('Образ распознан!')
                break
            prev = next


    @staticmethod
    def norm(matrix):
        for i in range(len(matrix)):
            if matrix[i][0] > 1:
                matrix[i][0] = 1
            elif matrix[i][0] < -1:
                matrix[i][0] = -1  
        return matrix          


if __name__=='__main__':
   Hp = Hopfield()
