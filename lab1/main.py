import matplotlib.image
import matplotlib.pyplot
from my_matrix_calculator import MatrixCalc as MC
from random import uniform
import copy
import math
import numpy as np


class Educate:
    def __init__(self):
        self.enter_parameters()
        self.hangle_pic()
        self.cicle_of_education()
    

    def enter_parameters(self):
        self.n = int(input('Введите размер n:\n'))
        self.m = int(input('Введите размер m:\n'))
        self.p = int(input(f'Введите число нейронов второго слоя p (p<={2*3*self.n*self.m}):\n'))
        self.e = int(input(f'Введите максимально допустимую ошибку e (e<{0.1*self.p})\n'))

        self.weights = [[uniform(-1,1) for j in range(self.p)] for i in range(self.m*self.n*3)]
        self.second_weights = MC.matrix_T(self.weights)
        self.koeff_alpha = 0.001


    def hangle_pic(self):
        img = matplotlib.image.imread('cat.png')
        self.img = MC.matrix_normalize(img)
        self.blocks = MC.reshape(self.img, self.m*3, self.n)
        print('кол-во блоков ', len(self.blocks))


    def cicle_of_education(self):
        E = 1e6
        i = 0
        while E > self.e:
            E = 0
            for block in self.blocks:
                E += self.train(block)
            i += 1
            print("i: ", i, '  E = ', E)
        self.save_weight()


    def train(self, block):
        block = [block]
        Y = MC.matrix_mult(block, self.weights)
        block_trait = MC.matrix_mult(Y, self.second_weights)
        delta_block = MC.matrix_diff(block, block_trait)
        stage_1_1 = MC.matrix_mult_number(MC.matrix_T(Y), self.koeff_alpha)
        stage_1_2 = MC.matrix_mult(stage_1_1, delta_block)
        self.second_weights = MC.matrix_diff(self.second_weights, stage_1_2)

        stage_2_1 = MC.matrix_mult_number(MC.matrix_T(block), self.koeff_alpha)
        stage_2_2 = MC.matrix_mult(stage_2_1, delta_block)
        stage_2_3 = MC.matrix_mult(stage_2_2, MC.matrix_T(self.second_weights))
        self.weights = MC.matrix_diff(self.weights, stage_2_3)
        self.normalization_weight()
        E = MC.matrix_sum(MC.matrix_square(delta_block))
        Y.clear()
        block_trait.clear()
        delta_block.clear()
        stage_1_1.clear()
        stage_1_2.clear()
        stage_2_1.clear()
        stage_2_2.clear()
        stage_2_3.clear()
        return E


    def normalization_weight(self):
        #нормализация весов 1
        w1_T = MC.matrix_T(self.weights)
        for col in range(len(self.weights[0])):
            module = MC.vector_module(w1_T[col])
            for row in range(len(self.weights)):
                self.weights[row][col] /= module

        #нормализация весов 2
        w2_T = MC.matrix_T(self.second_weights)
        for row in range(len(self.second_weights)):
            module = MC.vector_module(w2_T[row])
            for col in range(len(self.second_weights[0])):
                self.second_weights[row][col] /= module


    def save_weight(self):
        print("Обучение завершено. Сохранение весов.")

        with open('weights.txt', 'w') as f:
            for row in self.weights:
                f.write(str(row).strip(']').strip('[').replace(' ','')+'\n')

        with open('second_weights.txt', 'w') as f:
            for row in self.second_weights:
                f.write(str(row).strip(']').strip('[').replace(' ','')+'\n')

        with open("about_block.txt", "w") as file:
            file.write(str(self.n) + " " + str(self.m) + " " + str(self.hidden_layer_size))


class Use:
    def __init__(self):
        choice = int(input('Выберите опцию:\n1-распаковка файла\n2-сжатие файла\n'))
        file_name = input('Введите название файла:\n')
        self.load_educational_info()
        if choice ==1:
            new_img = open(file_name+".npy")

            print(self.image_unzip(new_img))
        if choice ==2:
            img = matplotlib.image.imread(file_name)
            self.img = MC.matrix_normalize(img)
            color_count = img.shape[0] * img.shape[1] * img.shape[2]
            print(self.image_zip(color_count))


    def image_zip(self, color_count):
        blocks = self.img.reshape((color_count // int(self.n* self.m * 3), int(self.n * self.m * 3)))
        new_img =[block @ self.weights for block in blocks]
        new_img = new_img.flatten()

        original_size = new_img.shape[0]
        
        while new_img.shape[0] % 3 != 0 or int(math.sqrt(new_img.shape[0] / 3)) != math.sqrt(new_img.shape[0] / 3):
            new_img.append(0)
        size = int(math.sqrt(new_img.shape[0] / 3))
        new_img = self.to_image(new_img, size, size)
        matplotlib.pyplot.imshow(new_img)
        matplotlib.pyplot.show()
        
        file_name = input("Сохранение сжатого файла. \nВведите название файла:\n")
        save_img = Educate.normalize(new_img).flatten()
        save_img.append([original_size])
        with open(file_name, 'w') as f:
            for row in save_img:
                f.write(str(row).strip(']').strip('[').replace(' ','')+'\n')
        print('Ok')


    def image_unzip(self, new_img):
        original_size = int(new_img[-1])
        new_img.pop()
        while new_img.shape[0] != original_size:
            new_img.pop()
        blocks = new_img.reshape((new_img.shape[0] // self.p, self.p))
        new_img = [block @ self.decode_weights for block in blocks]
        new_img = new_img.flatten()
        size = int(math.sqrt(new_img.shape[0] / 3))
        new_img = self.to_image(new_img, size, size)
        matplotlib.pyplot.imshow(new_img)
        matplotlib.pyplot.show()
        file_name = input("Сохранение картинки. \nВведите название файла:\n")
        matplotlib.pyplot.imsave(file_name, new_img)
        return 'Ok'



    def to_image(self, colors, size1, size2):
        return Use.denormalize(MC.reshape(colors, size1, size2, 3))
    
    
    def denormalize_range(x):
        x = (x + 1) / 2
        if x > 1: 
            x = 1
        if x < 0:
            x = 0
        return x 


    def denormalize(pixels):
        pixels = [[[ Use.denormalize_range(pixels[i][j][color])  for color in range(len(pixels[i][j]))] for j in range(len(pixels[i]))] for i in range(len(pixels))]
        return pixels


    def load_educational_info(self):
        self.weights = np.load(r"weights.npy")
        self.decode_weights = np.load(r"second_weights.npy")
        with open(r"about_block.txt", "r") as file:
            data = file.read().split()
            self.n = int(data[0])
            self.m = int(data[1])
            self.p = int(data[2])
        return 'Ok'
            



if __name__=='__main__':
    choice = int(input('Выберите режим: \n1- обучение\n0- использование сети\n'))
    if choice == 1:
        educate = Educate()
    if choice == 2:
        use = Use()



