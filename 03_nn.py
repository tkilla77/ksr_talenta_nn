import numpy as np
 
# ACTIVATION FUNCTION
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
# NETWORK
class Network:
    def __init__(self, N_input, N_hidden, N_output,learning_rate=0.02):
        self.N_input = N_input
        self.N_hidden = N_hidden
        self.N_output = N_output
        self.learning_rate = learning_rate
 
        self.wA = np.random.rand(self.N_hidden, self.N_input) - 0.5
        self.wB = np.random.rand(self.N_output, self.N_hidden) - 0.5
 
    def read_data(self,record):
        """
        this method takes a record (1 line of csv data file) as string and returns the input and target output vectors
        """
        data = record.split(',')
        # INPUT DATA
        x = np.asfarray(data[1:len(data)]) / 255.  # divide by 255 to get nr btw 0 and 1
        # TARGET DATA
        target_nr = int(data[0]) #number
        # convert number into vector, with 1 at position corresponding to number
        target_vec = np.zeros(self.N_output)
        target_vec[target_nr] = 1
        # reshape from (n,) array to (n,1) array. Otherwise run into problem when updating weight matrices
        x = x.reshape((-1,1))
        target_vec = target_vec.reshape((-1,1))
        return x,target_vec
 
    def feedforward(self, x):
        """
        given a data point (input data x), feed forward to determine output y for current weight matrices
        """
        h = sigmoid(np.dot(self.wA, x))
        y = sigmoid(np.dot(self.wB, h))
        return h, y
 
    def train_single_record(self,x,t):
        """
        train network (i.e. update weight matrices) for a single data point
        """
        h,y = self.feedforward(x)
 
        # COST
        # target output - actual output
        print(t.shape, y.shape)
        error_output = t - y
        print(error_output.shape)
 

        # SPLIT ERROR
        # proportional to weighs
        error_hidden = np.dot(self.wB.T,error_output)
        error_input = np.dot(self.wA.T,error_hidden)
 
        # UPDATE WEIGHTS
        self.wB += np.dot(self.learning_rate*error_output*y*(1-y),h.T)
        self.wA += np.dot(self.learning_rate*error_hidden*h*(1-h),x.T)

        return error_output
 
    def train(self,data_list):
        """
        Train network for list of data points.
        """
        count = 0
        for record in data_list:
            x,t = self.read_data(record)
            # train network
            count += 1
            error = self.train_single_record(x,t)
            if count % 1000 == 1:
                print(f"Error: {np.linalg.norm(error)}")
 
    def test(self, data_list):
        """
        test neural network, return rate of how successful
        """
        count_correct = 0
        count_wrong = 0
        for record in data_list:
            x,t = self.read_data(record) # x is input vector, t is target vector
 
            h, y = self.feedforward(x)
 
            i_max = np.argmax(abs(y)) # get index of max value in output
            if t[i_max] == 1:
                count_correct += 1
            else:
                count_wrong += 1      
 
        ratio = count_correct/(count_correct+count_wrong)
        print(f"success: {ratio*100}%")
        return ratio 
 
# ------------------------------------------------------------
 
# CREATE NETWORK OBJECT
# 1. network: dark bright
def dark_bright():
    print("NEURAL NETWORT DARK BRIGHT")
    with open('data_csv/data_dark_bright_training_20000.csv', 'r') as f:
        data_db_train_list = f.readlines()
    with open('data_csv/data_dark_bright_test_4000.csv', 'r') as f:
        data_db_test_list = f.readlines()
    
    network_db = Network(4, 3, 2)
    
    print("Test before training:")
    network_db.test(data_db_test_list)
    print("Train neural network:")
    network_db.train(data_db_train_list) 
    print("Test after training:")
    network_db.test(data_db_test_list)
 
    print("-------------------")
 
# 2. network: mnist data
def mnist():
    print("NEURAL NETWORT MNIST")
    with open('data_mnist/mnist_train.csv', 'r') as f:
        data_mnist_train_list = f.readlines()
    with open('data_mnist/mnist_test.csv', 'r') as f:
        data_mnist_test_list = f.readlines()
    
    network_mnist = Network(784, 30, 10)
    
    print("Test before training:")
    network_mnist.test(data_mnist_test_list)
    print("Train neural network:")
    network_mnist.train(data_mnist_train_list)
    print("Test after training:")
    network_mnist.test(data_mnist_test_list)

mnist()