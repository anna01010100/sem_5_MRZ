import copy

class MatrixCalc:

    def get_shape(a):
        m = len(a)
        if m == 0:
            raise Exception('Empty matrix')
        if type(a[0])!= list:
            return (m, 1)
        else: 
            n = len(a[0])
            if type(a[0][0]) != list:
                return(m, n)
            else:
                k = len(a[0][0])
                return(m,n,k)


    def matrix_mult(a, b):
        m, n1, *high = MatrixCalc.get_shape(a)
        n2, k, *high2 = MatrixCalc.get_shape(b)
        if n1 != n2:
            raise Exception('Cannot multiply these matrix')                                                    
        c = [[sum(a[i][kk] * b[kk][j] for kk in range(n1)) for j in range(k)] for i in range(m)]    
        return c


    def matrix_mult_number(a, value):
            for i in range(len(a)):
                for j in range(len(a[0])):
                    a[i][j] *= value
            return a


    def matrix_T(a):
        T = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
        return T


    def matrix_diff(a, b):
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise 
        result = [[a[i][j]-b[i][j] for j in range(len(a[0]))]for i in range(len(a))]
        return result


    def matrix_normalize(a): 
        a = copy.deepcopy(a)
        return [[[color*2 -1 for color in j] for j in i]for i in a]


    def vector_diff(v1, v2):
            if len(v1) != len(v2):
                raise
            result = [v1[i]-v2[i] for i in range(len(v1))]
            return result


    def reshape(a, n, m):
        block_size = n*m
        shape = MatrixCalc.get_shape(a)
        all_cells = shape[0]*shape[1]*shape[2]
        a = [i for n in a for m in n for i in m]
        count_rows = block_size
        count_cols = all_cells // block_size
        result = []
        index = 0
        for i in range(count_cols):
            result.append([])
            for j in range(count_rows):
                result[i].append(a[index])
                index += 1
        return result


    def vector_module(v):
        #return [list(map(lambda x: x*n, z)) for z in lst]
        sum_of_squares = 0
        return float(sum((map(lambda x: x **2, v))) ** 0.5)


    def matrix_square(m):
        return [[i**2 for i in j] for j in m]


    def matrix_sum(m):
        return float(sum(sum(i) for i in m))


A = [3,4,5]

print(MatrixCalc.vector_module(A))
