import numpy as np

'''
    creating a class called dlFunctions which would contain all the functions required
'''
class dlFunctions:
    def __init__(self):
        pass
    
    # defining a helping function for calculating max while using the relu
    def max_relu(self,x):
        return max(0,x)

    def forw_relu(self, x : np.ndarray):
        # asserting that the data type of each element in the array x is float. This is followed in rest of the functions as well
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        maximize = np.vectorize(self.max_relu)
        return maximize(x)

    def back_relu(self, x : np.ndarray, y: np.ndarray, dzdy: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(y)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(dzdy)).dtype == 'float', "Please enter a floating point numpy array"

        # defining a mask array
        bool_array = np.zeros((x.shape[0],x.shape[1]))
        # setting the mask value as 1 if the value of x is greater or equal to 0
        bool_array[np.where(x >= 0.0)] = 1
        # multiplying our received derivatives from the next network layers with our mask
        dzdx = dzdy * bool_array
        
        return dzdx

    '''
        Assuming the filter size is 2X2
    '''
    def forw_maxpool(self, x: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"

        width = x.shape[1]
        height = x.shape[0]
        y = np.zeros((height//2, width//2))

        for i in range(0,height,2):
            for j in range(0,width,2):
                y[i//2][j//2] = np.max(x[i:i+2,j:j+2])
        return y

    def back_maxpool(self, x: np.ndarray, y: np.ndarray, dzdy: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(y)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(dzdy)).dtype == 'float', "Please enter a floating point numpy array"

        bool_array = np.zeros((x.shape[0],x.shape[1]))
        dzdx = np.zeros((x.shape[0],x.shape[1]))
        width = x.shape[1]
        height = x.shape[0]

        for i in range(0,height,2):
            for j in range(0,width,2):
                bool_array[i:i+2,j:j+2] = (x[i:i+2,j:j+2] == np.max(x[i:i+2,j:j+2]))
                dzdx[i:i+2,j:j+2] = bool_array[i:i+2,j:j+2] * dzdy[i//2,j//2]
        
        return dzdx

    def forw_meanpool(self, x: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"

        width = x.shape[1]
        height = x.shape[0]
        y = np.zeros((height//2, width//2))

        for i in range(0,height,2):
            for j in range(0,width,2):
                y[i//2][j//2] = np.mean(x[i:i+2,j:j+2])
        return y

    def back_meanpool(self, x: np.ndarray, y: np.ndarray, dzdy: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(y)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(dzdy)).dtype == 'float', "Please enter a floating point numpy array"

        dzdx = np.zeros((x.shape[0],x.shape[1]))
        width = x.shape[1]
        height = x.shape[0]

        for i in range(0,height,2):
            for j in range(0,width,2):
                dzdx[i:i+2,j:j+2] = 1/4 * dzdx[i//2,j//2]
        
        return dzdx

    def forw_fc(self, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(w)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(b)).dtype == 'float', "Please enter a floating point numpy array"

        y = np.sum(x * w) + b
        return y

    def back_fc(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, y: np.ndarray, dzdy: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(w)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(b)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(y)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(dzdy)).dtype == 'float', "Please enter a floating point numpy array"
        
        dzdb = np.array([dzdy]).reshape(1,1)
        dzdx = dzdy * b
        dzdw = dzdy * x

        return dzdx, dzdw, dzdb


    def forw_softmax(self, x: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"

        y = np.exp(x)/np.sum(np.exp(x))
        return y

    def back_softmax(self, x: np.ndarray, y: np.ndarray, dzdy: np.ndarray):
        assert np.array(list(x)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(y)).dtype == 'float', "Please enter a floating point numpy array"
        assert np.array(list(dzdy)).dtype == 'float', "Please enter a floating point numpy array"

        complete_matrix = -1 * y * y.reshape(1,y.shape[0]) 
        dzdx = complete_matrix + np.diag(y.reshape(y.shape[0],))
        dzdx = dzdx * dzdy
        return dzdx
