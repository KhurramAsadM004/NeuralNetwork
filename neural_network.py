import numpy as np
import matplotlib.pyplot as plt
import f_utils
#import copy
from f_check_gradient import check_gradients
#from sklearn.utils import shuffle
import seaborn as sb
#import sys

plt.ion()


class NeuralNetwork():
  
    def __init__(self, args):     
        self.num_neurons = args.layer_dim
        self.activations_func = args.activations
        self.num_layers = len(self.num_neurons) - 1 
        self.check_grad = args.check_grad
        if self.check_grad == True:
            self.grad_ok = 0
        else:
            self.grad_ok = 1
                
        self.epochs = args.epochs
        self.convergence_threshold = args.convergence_threshold
        self.mini_batch_size = args.batch_size          
        self.learning_rate = args.learning_rate
        self.optimizer = args.optimizer        
        self.loss = args.loss
        self.early_stopping = args.early_stopping        
        self.patience = args.patience
        
        # ADAM hyperparameters from original paper https://arxiv.org/pdf/1412.6980.pdf
        self.alpha = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8 
        
        self.weights_save_dir = args.weights_save_dir        
        self.mode = args.mode
        
        if self.mode == 'test':
            self.arch_filename = self.weights_save_dir + args.arch_file
            self.weights_filename = self.weights_save_dir + args.weights_file

        '''
        self.parameters is a dictionary that will store the weights and biases
        self.parameters['W1'] will store the weights of the 1st layer
        self.parameters['b1'] will store the biases of the 1st layer
        self.parameters['W2'] will store the weights of the 2nd layer
        self.parameters['b2'] will store the biases of the 2nd layer
        and so on
        '''
        self.parameters = dict()
        
        '''
        self.net is a dictionary that will store the outputs of every neuron
        self.net['z0'] will contain the output of the input layer aka the input
        self.net['z1'] will contain the output of the 1st layer
        and so on
        
        if layer z1 has 5 neurons and batch size is 10, then size of z1 will
        be [5,10] and similarly for other layers
        '''
        self.net = dict()
        
        '''
        self.grads is a dictionary that will store the gradients
        self.grads['dW1'] will store the gradients for weights of the 1st layer
        self.grads['db1'] will store the gradients for biases of the 1st layer
        self.grads['delta1'] will store the delta values of neurons in the 1st layer
        self.grads['dW2'] will store the gradients for weights of the 2nd layer
        self.grads['db2'] will store the gradients for biases of the 2nd layer
        self.grads['delta2'] will store the delta values of neurons in the 2nd layer
        and so on
        '''
        self.grads = dict()
       
        print("Training parameters:")
        print(f"num_neurons: {self.num_neurons}")
        print(f"Activation functions: {self.activations_func}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mini batch size: {self.mini_batch_size}")
        print(f"Epochs: {self.epochs}")   
        print(f"Check gradient: {self.check_grad}") 
        print(f"Early stopping: {self.early_stopping}")
        print("*"*50)
        
    def initialize_parameters(self):
         # Set the random seed for reproducibility
         np.random.seed(45)
         print("Initializing neural network weights and biases for training...")
         
         # Iterate over each layer
         for l in range(1, self.num_layers + 1):
             
             # Initialize weights using Gaussian distribution with mean 0 and variance 1
             self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1])
             
             # Pick appropriate value of variance
             if self.activations_func[l] == 'relu' or self.activations_func[l] == 'lrelu':
                 # For ReLU and variants, use Xavier's intialization which uses variance = num_neurons[l-1]/2
                 self.parameters['W%s' % l] /= np.sqrt(self.num_neurons[l-1]/2.)
             else:                
                 # Use standard initialization which uses variance = num_neurons[l-1]
                 self.parameters['W%s' % l] /= np.sqrt(self.num_neurons[l - 1])
             
             # Initialize biases to zero
             self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
             
             print(f"Sizes of weights and biases initialized at layer {l}:",self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)

         ### for optimizers
         if self.optimizer == 'adam':          
            self.m, self.v = f_utils.initialize_adam(self)

        
    def fprop(self, batch_input):
       # Set the input of the first layer to the batch_input
       self.net['z0'] = batch_input
       # Iterate over each layer
       for l in range(1, self.num_layers+1):
           # Retrieve the weights and biases for the current layer
           W = '''ADD CODE HERE'''
           b = '''ADD CODE HERE'''
           
           # Retrieve the function in f_utils corresponding to the string in self.activations_func[l]
           activation_func = getattr(f_utils,self.activations_func[l])
           
           # Retrieve the output of the previous layer
           x = '''ADD CODE HERE'''
           
           # Compute the pre-activation value a = Wx+b
           a = '''ADD CODE HERE'''
           
           # Apply the activation function
           z = activation_func(a)
           
           # Store the output of the current layer in self.net dictionary             
           self.net['z%s' % l] = z
   
                

            
    def calculate_loss(self, batch_target):
        y = self.net['z%s' % str(self.num_layers)]    

        if self.loss == 'mse':
            loss = f_utils.mse(self, y, batch_target)
        elif self.loss == 'mce':
            loss = f_utils.mce(self, y, batch_target) 
        elif self.loss == 'bce':
            loss = f_utils.bce(self, y, batch_target) 
        return loss
    
    def update_parameters(self, t):    
        if self.optimizer == 'sgd':                    
            for l in range(1, self.num_layers+1): # Iterate over each layer
                # Update the weights of the current layer
                self.parameters['W%s' % l] -= '''ADD CODE HERE'''
                
                # Update the biases of the current layer
                self.parameters['b%s' % l] -= '''ADD CODE HERE'''
          
        elif self.optimizer == 'adam':
            t = t+1
            for l in range(1, self.num_layers+1):
                '''
                Implement the while-loop in Algorithm 1 of https://arxiv.org/pdf/1412.6980.pdf
                '''
                # Moving average of the gradients: m = beta1*m + (1-beta1)*dL/dw
                self.m["dW%s" % l] = self.beta1*self.m["dW%s" % l] + '''ADD CODE HERE'''   #for weights
                self.m["db%s" % l] = '''ADD CODE HERE'''    #for biases

                # Compute bias-corrected first moment estimate: m_corrected = m / (1-beta1^t)
                m_corrected_W = '''ADD CODE HERE'''    #for weights
                m_corrected_b = '''ADD CODE HERE'''    #for biases
      
                # Moving average of the squared gradients: v = beta2*v + (1-beta2)*(dL/dw)^2
                self.v["dW%s" % l] = '''ADD CODE HERE'''    #for weights
                self.v["db%s" % l] = '''ADD CODE HERE'''    #for biases
       
                # Compute bias-corrected second moment estimate: v_corrected = v / (1-beta2^t)
                v_corrected_W = '''ADD CODE HERE'''    #for weights
                v_corrected_b = '''ADD CODE HERE'''    #for biases

                # Update parameters: w = w - alpha * m_corrected/(sqrt(v_corrected)+eps)
                self.parameters["W%s" % l] = '''ADD CODE HERE'''    #for weights
                self.parameters["b%s" % l] = '''ADD CODE HERE'''    #for biases


           
    
    def bprop(self, batch_target):
        batch_size = batch_target[0].shape
        
        # Retrieve output of the last layer of the neural network
        y = '''ADD CODE HERE'''
        
        # Compute delta_k values for these output layer neurons
        delta_k = '''ADD CODE HERE'''
        
        # Retrieve outputs of previous layer. These are the inputs to the output layer.
        z = '''ADD CODE HERE'''
        
        # Compute gradients of weights via delta_k x input
        # If using MSE loss, do not forget to take mean over the mini-batch
        # Size of dW should be the same as size of the weight matrix W of the output layer
        dW = '''ADD CODE HERE'''
        
        # Compute gradients of biases via delta_k x 1
        # If using MSE loss, do not forget to take mean over the mini-batch
        # Size of db should be the same as the size of the bias vector b of the output layer
        db = '''ADD CODE HERE'''
        
        # Store gradients for the output layer in self.grads dictionary
        self.grads['dW%s' % str(self.num_layers)] = dW 
        self.grads['db%s' % str(self.num_layers)] = db
        self.grads['delta%s' % str(self.num_layers)] = delta_k
        
        # Now start back-propagating the delta values through the previous layers one-by-one
        for l in reversed(range(1, self.num_layers)):
            # At this point, variable z already contains the outputs of current layer l
            
            # Retrieve the derivative function in f_utils corresponding to the string in self.activations_func[l]
            activation_derivative_func = getattr(f_utils, self.activations_func[l]+'_derivative')
            
            # Calculate the derivative of the activation function using outputs in z
            hprime = activation_derivative_func(z);
                       
            # Compute delta values for current layer using the back-propagation equation
            delta_j = np.multiply(hprime, np.dot('''ADD CODE HERE'''))
            
            # Retrieve outputs of previous layer
            # save them in z for this as well as the next iteration
            z = '''ADD CODE HERE'''
            
            # Compute gradients of weights via delta_j x input
            # If using MSE loss, do not forget to take mean over the mini-batch
            # Size of dW should be the same as size of the weight matrix W of layer l
            dW =  '''ADD CODE HERE'''
            
            # Compute gradients of biases via delta_j x 1
            # If using MSE loss, do not forget to take mean over the mini-batch
            # Size of db should be the same as size of the bias vector b of layer l
            db =  '''ADD CODE HERE'''
            
            # Store gradients for current layer in self.grads dictionary
            self.grads['dW%s' % l] = dW
            self.grads['db%s' % l] = db
            self.grads['delta%s' % l] = delta_j
       
           


    def train(self, train_x, train_t, val_x, val_t):      
        # Initialize model parameters    
        print(train_x.shape, train_t.shape, val_x.shape, val_t.shape)
        
        self.initialize_parameters()
        
        # Create lists to store training and validation losses
        train_loss_l = [] # List to store training loss
        val_loss_l = []   # List to store validation loss

        # Retrieve the number of training samples
        num_samples = train_t.shape[1]
        
        ## set values for early stopping 
        best_val_loss = 100
        count = 0  
        
        t=0
        
        # Start training epochs
        print("Training started....")
        # Iterate over the specified number of epochs
        for i in range(0, self.epochs):
            train_loss_b = []
            # Iterate over training samples in mini-batches
            for idx in range(0, num_samples, self.mini_batch_size):
                # Get mini-batch input
                minibatch_input =  train_x[:, idx:idx + self.mini_batch_size]
                # Get mini-batch target
                minibatch_target =  train_t[:, idx:idx + self.mini_batch_size]
              
                if self.grad_ok == 0:
                    # Check gradients
                    self.grad_ok = check_gradients(self, minibatch_input, minibatch_target)
                    if self.grad_ok  == 0:
                        print("Gradients are not ok! There is no point in training the network.\n")
                        return self.grad_ok
                   
                if self.grad_ok == 1:                   
                    # For current mini-batch, perform fprop and bprop and then update the parameters 
                    self.fprop(minibatch_input)
                    loss = self.calculate_loss(minibatch_target)
                    train_loss_b.append(loss)
                    self.bprop(minibatch_target)           
                    self.update_parameters(t)
                    
              
            '''
            Training over all mini-batches is now complete for current epoch
            Now compute losses over complete training and validation sets
            '''
            #Compute training loss after current epoch
            train_loss = np.mean(train_loss_b)
            train_loss_l.append(train_loss)
            
            #Compute validation loss after current epoch
            self.fprop(val_x)
            val_loss = self.calculate_loss(val_t)
            val_loss_l.append(val_loss) 
            
           
            print("Epoch %i: training loss %f, validation loss %f" % (i, train_loss, val_loss))
            
            plt.subplot(2,1,1)
            plt.plot(np.round(train_loss_l, 2), color='g', linewidth=3, label="train")
            plt.plot(np.round(val_loss_l, 2), color='m', linewidth=3, label="val")
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.title(f'Optimizer: {self.optimizer}, Learning rate: {self.learning_rate}') # Set the title of the plot 
            watermark_text = 'Official Solution'
            plt.text(0.5, 0.5, watermark_text, fontsize=20, color='gray', ha='center', va='center', alpha=0.5, transform=plt.gca().transAxes)
            plt.grid() 
            plt.legend()
            
            plt.subplots_adjust(hspace=0.5)
            
            plt.subplot(2,1,2)            
            train_acc, confusion_matrix, _ = self.test(train_x, train_t, 'train')  
            val_acc, _, _ = self.test(val_x, val_t, 'train')  
            plt.title(f'Confusion matrix | Train acc. {np.round(train_acc,2)} | Val acc. {np.round(val_acc,2)}') # Set the title of the plot 
            classes = [str(i) for i in range (0,self.num_neurons[-1])]
            sb.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.draw()
            plt.pause(0.01)
            plt.clf()
            
            # check convergence  
            if i > 0 and abs(train_loss - train_loss_l[i-1]) < self.convergence_threshold:
                   print("Converged!!")
                   break
               
            # early stopping
            if self.early_stopping == True:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    count = 0
                else:
                    count += 1
                
                if count >= self.patience:
                    print(f"Training stopped. No improvement in validation loss for {self.patience} consecutive epochs...")
                    break
            
                
        print("Training completed.\n")
        plt.close()
        f_utils.plot_loss(self, train_loss_l, val_loss_l)   # Plot training and validation loss
        f_utils.save_loss(self, train_loss_l, val_loss_l)   # Save training and validation losses
        # plot_weights(self)                          # Plot model weights
        # plot_gradients(self)                        # Plot gradients
        f_utils.save_model(self)                    # save model details and weights
    
    def test(self, test_X, test_t, mode='test'):
            
            # load model details and weights if in test mode
            if mode == 'test':
                f_utils.load_model(self)
            
            # number of testing examples
            size = test_t.shape[1]    
            
            # Initialize count and confusion matrix
            count = 0
            confusion_matrix = np.zeros((self.num_neurons[-1],self.num_neurons[-1]))
            
            # Perform forward propagation on the test data
            self.fprop(test_X)
            
            # Calculate the loss using the test targets
            loss = self.calculate_loss(test_t)
            
            # Retrieve the output of the neural network
            output = self.net['z%s' % str(self.num_layers)]             
          
    
            # Iterate over all testing examples
            for i in range(size):
                # Get the predicted class index
                y = np.argmax(output[:,i])
                
                # Get the true class index
                t = np.argmax(test_t[:,i])               

                # Update the count and confusion matrix based on correct and incorrect predictions
                # Update appropriate entry of confusion matrix based on predicted and true class
                confusion_matrix['''ADD CODE HERE'''] += 1

            
            # Calculate the accuracy 
            number_of_correct_predictions = '''ADD CODE HERE''' #HINT: Diagonal of confusion matrix
            total_predictions = '''ADD CODE HERE'''
            accuracy = (number_of_correct_predictions/total_predictions)*100
            
            # Return the calculated accuracy and the confusion matrix
            return accuracy, confusion_matrix.astype(int), self.optimizer
