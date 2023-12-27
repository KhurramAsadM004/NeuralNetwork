import numpy as np
from neural_network import *
from f_load_data import *
import argparse
import matplotlib.pyplot as plt
from os.path import isfile

parser = argparse.ArgumentParser(description='')
parser.add_argument('--layer_dim', dest='layer_dim', type=int, nargs='+', default=[784, 50, 20, 10], help='no of layers with neurons')
parser.add_argument('--activations', dest='activations', type=str, nargs='+', default=[None, 'relu', 'relu', 'softmax'], help='activation function')
parser.add_argument('--optimizer', dest='optimizer', default='adam', choices=['sgd', 'adam'], help='optimization algorithm')
parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='epochs')
parser.add_argument('--loss', dest='loss', default='mce', choices=['mce'], help='loss functions')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='batch_size: 48, 64, 128,..')
parser.add_argument('--check_grad', dest='check_grad', action='store_true', help='check gradient ..')
parser.add_argument('--early_stopping', dest='early_stopping', action='store_true', help='Enable early stopping. If set True, training will stop when validation loss stops improving for a certain number of epochs.')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-2, help='learning_rate')
parser.add_argument('--patience', dest='patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped.')
parser.add_argument('--convergence_threshold', dest='convergence_threshold', type=float, default=1e-5, help='')
parser.add_argument('--mode', dest='mode', type=str, default='test', choices=['train', 'test'], help='')
parser.add_argument('--weights_save_dir', dest='weights_save_dir', type=str, default='', help='path')
parser.add_argument('--arch_file', dest='arch_file', type=str, default='model_details.json', help='json file')
parser.add_argument('--weights_file', dest='weights_file', type=str, default='model_weights.npy', help='npy file')
parser.add_argument('--data_dir', dest='data_dir', type=str, default='../Data/', help='')
args = parser.parse_args()

nn = NeuralNetwork(args)
# load dataset
train_x, train_t, val_x, val_t, test_x, test_t = load_dataset(args)
print("train_x and train_t: ", train_x.shape, train_t.shape)
print("val_x and val_t: ", val_x.shape, val_t.shape)
print("test_x and test_t: ", test_x.shape, test_t.shape)

if args.mode == 'train':
    nn.train(train_x, train_t, val_x, val_t)
else:
    test_acc, confusion_matrix, optimizer = nn.test(test_x, test_t) 
    print("testing acc..", np.round(test_acc,2))
    print("confusion_matrix: \n", confusion_matrix)
    
    '''
    Plot and save confusion matrix on test data
    '''
    plt.title(f'Confusion matrix | Test acc. {np.round(test_acc,2)}') # Set the title of the plot 
    classes = [str(i) for i in range (0,test_t.shape[0])]
    sb.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.draw()
    plt.savefig("ConfusionMatrixTest_"+optimizer+".png")
    plt.clf()
    
    '''
    Plot training losses of SGD and ADAM models on the same plot
    '''
    if isfile('adam_losses.npz') and isfile('sgd_losses.npz'):
        data = np.load('adam_losses.npz')
        adam_train_losses = data['train_losses']
        adam_val_losses = data['val_losses']
        
        data = np.load('sgd_losses.npz')
        sgd_train_losses = data['train_losses']
        sgd_val_losses = data['val_losses']
        
        plt.figure() 
        fig = plt.gcf() 
        plt.plot(adam_train_losses, linewidth=3, label="ADAM: Training") # Plot the training loss with a line width of 3 and label="train"
        plt.plot(adam_val_losses, linewidth=3, label="ADAM: Validation")  # Plot the validation loss with a line width of 3 and label="val"
        plt.plot(sgd_train_losses, linewidth=3, label="SGD: Training") # Plot the training loss with a line width of 3 and label="train"
        plt.plot(sgd_val_losses, linewidth=3, label="SGD: Validation")  # Plot the validation loss with a line width of 3 and label="val"
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('ADAM vs. SGD') # Set the title of the plot 
        plt.grid() 
        plt.legend()
        plt.show() 
        fig.savefig('adam_vs_sgd.png')  
        plt.close()








