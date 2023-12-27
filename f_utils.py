import numpy as np
import json
import matplotlib.pyplot as plt

def normalize_data(data):   
    """
    This function calculates the mean and variance of each image in the input data and uses them to
    standardize the data. It ensures that each feature has a mean of 0 and a standard deviation of 1.
    """
    eps=1e-8
    mean = np.mean(data, axis=1, keepdims=True)
    variance = np.var(data, axis=1, keepdims=True)
    data_norm = np.divide((data - mean), np.sqrt(variance+eps))
    return data_norm

def sigmoid(a):
    """
    Computes the sigmoid function element-wise on 'a'.  
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the sigmoid activation function.    
    """
    z =  1/(1 + np.exp(-a))
    return z

def tanh(a):
    """
    Computes hyperbolic tangent (tanh) function element-wise on 'a'.  
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the tanh activation function.    
    """
    z = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
    return z

def relu(a):
    """
    Computes the rectified linear unit (ReLU) function element-wise on 'a'.    
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the relu activation function.    
    """
    z = np.where(a > 0, a, 0)
    return z
  

def lrelu(a, k=0.01):
    """
    Computes the leaky rectified linear unit (Leaky ReLU) function element-wise on 'a' with a given slope 'k'.
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the lrelu activation function.    
    """
    z = np.where(a > 0, a, a * k)
    return z

def identity(a):
    return a

def sigmoid_derivative(z):
    """
    Computes the derivative of the sigmoid function element-wise on 'z'.   
    'hprime' contains sigmoid function derivative values.    
    """
    hprime = z * (1 - z)
    return hprime

def tanh_derivative(z):
    """
    Computes the derivative of the hyperbolic tangent (tanh) function element-wise on 'z'. 
    'hprime' contains tanh function derivative values. 
    """
    hprime = 1 - (z**2)
    return hprime

def relu_derivative(z): 
    """
    Computes the derivative of the rectified linear unit (ReLU) function element-wise on 'z'.
    'hprime' contains relu function derivative values. 
    """
    hprime = np.where(z > 0, 1, 0)
    return hprime

def lrelu_derivative(z, k=0.01): 
    """
    Computes the derivative of the leaky rectified linear unit (Leaky ReLU) function element-wise on 'z'
    with a given slope 'k'.   
    'hprime' contains lrelu function derivative values. 
    """    
    hprime = np.where(z > 0, 1, k)
    return hprime

def identity_derivative(z):
    return np.ones_like(z)


def softmax(a):
    max_a = np.amax(a, axis=0) #compute maximum of pre-activations in a (1 max per sample if using mini-batches)
    a_exp = np.exp(a - max_a) #exponentials of pre-activations after subtracting maximum
    a_sum = np.sum(a_exp, axis=0) + 1e-9 #sum of exponentials
    z = np.divide(a_exp, a_sum) #softmax results
    return z

def initialize_adam(self) :
    m = {}
    v = {}
    for l in range(1, self.num_layers+1):
        m["dW%s" % l] =  np.zeros(self.parameters["W%s" % l].shape)
        m["db%s" % l] =  np.zeros(self.parameters["b%s" % l].shape)
        v["dW%s" % l] =  np.zeros(self.parameters["W%s" % l].shape)
        v["db%s" % l] =  np.zeros(self.parameters["b%s" % l].shape)    
    return m, v

def mse(self, y, batch_target):
    loss = np.mean(np.square(y - batch_target)) #mean squared loss for regression
    return loss

def mce(self, y, batch_target):
    eps = 1e-9
    loss = -((1/y.shape[1]) * np.sum(np.sum(batch_target * np.log(y + eps)))) #mean multiclass cross-entropy loss for multiclass classification
    return loss

def bce(self, y, batch_target):
    eps = 1e-9
    loss = -np.mean(t * np.log(y + eps) + (1-t) * np.log((1-y) + eps), axis=0) #mean binary cross-entropy loss for binary classification
    return loss

def save_model(self):
    model_details = {
    'layer_dim': self.num_neurons,
    'activations': self.activations_func,
    'optimizer': self.optimizer,
    'epochs': self.epochs,
    'loss': self.loss,
    'batch_size': self.mini_batch_size,
    'learning_rate': self.learning_rate,
    'mode': self.mode,
    'weights_save_dir': self.weights_save_dir,
    'losses_filename': self.losses_filename
    }
    # open a file and use the json.dump method to save model details 
    self.arch_filename = self.weights_save_dir+self.optimizer+'_model.json'
    with open(self.arch_filename, "w") as json_file:
        json.dump(model_details, json_file)

    # save model weights as a numpy array in a .npy file
    self.weights_filename = self.weights_save_dir+self.optimizer+'_model.npy'
    np.save(self.weights_filename, self.parameters)
    
        
def load_model(self):
    with open(self.arch_filename, 'r') as json_file:
        model_details = json.load(json_file)
        
        self.num_neurons = model_details['layer_dim']   
        self.optimizer = model_details['optimizer']
        self.loss = model_details['loss']
        self.activations_func = model_details['activations'] 
        self.weights_save_dir = model_details['weights_save_dir']
        self.losses_filename = model_details['losses_filename']
        print(model_details)
        
    loaded_weights = np.load(self.weights_filename, allow_pickle=True)

    # The .item() method is used here to convert the numpy array into a dictionary-like structure
    # that was originally used to save and store model parameters.
    self.parameters = loaded_weights.item() 
    # print(self.parameters)



def plot_loss(self, loss, val_loss):        
    """
    Plot the training and validation loss curves and save figure    
    """
    plt.figure() 
    fig = plt.gcf() 
    plt.plot(loss, linewidth=3, label="Training") # Plot the training loss with a line width of 3 and label="train"
    plt.plot(val_loss, linewidth=3, label="Validation")  # Plot the validation loss with a line width of 3 and label="val"
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title(f'Optimizer: {self.optimizer}, Learning rate: {self.learning_rate}') # Set the title of the plot 
    plt.grid() 
    plt.legend()
    plt.show() 
    fig.savefig('loss_trainval_'+self.optimizer+'.png')  
    plt.close()

def save_loss(self, train_losses, val_losses):
   """
   Save training and validation losses
   """
   self.losses_filename = self.weights_save_dir+self.optimizer+'_losses.npz'
   np.savez(self.losses_filename, train_losses=train_losses, val_losses=val_losses)

def load_loss(self):
   """
   Save training and validation losses
   """
   data = np.load(self.losses_filename)
   train_losses = data['train_losses']
   val_losses = data['val_losses']
   return train_losses, val_losses
