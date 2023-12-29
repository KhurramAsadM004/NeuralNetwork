from f_utils import *
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io

def load_dataset(args):
    if(args.dataset == "mnist"):
        return load_dataset_mnist(args)
    elif (args.dataset == "cifar"):
        return load_dataset_cifar(args)
    else 
        print("No dataset provided.")
        exit()

def load_dataset_mnist(args):
    print("loading data.....")
    
    data_file = scipy.io.loadmat(args.data_dir+'mnist_uint8.mat')
    
    train_x = data_file['train_x']
    train_y = data_file['train_y']
    test_x = data_file['test_x']
    test_y = data_file['test_y']

    print("train_x, train_y, test_x, test_y",train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        
    train_x, val_x, train_t, val_t = train_test_split(train_x, train_y, test_size=0.2)

    train_x = normalize_data(train_x)
    val_x = normalize_data(val_x)
    test_x = normalize_data(test_x)   
    
    
    
    return train_x.T, train_t.T, val_x.T, val_t.T, test_x.T, test_y.T   

def load_dataset_cifar(args):
    print("loading data.....")
    
    data_file1 = unpickle(args.data_dir+'data_batch_1')
    data_file2 = unpickle(args.data_dir+'data_batch_2')
    data_file3 = unpickle(args.data_dir+'data_batch_3')
    data_file4 = unpickle(args.data_dir+'data_batch_4')
    data_file5 = unpickle(args.data_dir+'data_batch_5')
    test_file = unpickle(args.data_dir+'test_batch')
    
    # print(data_file.keys(), test_file.keys())
    train_x = np.concatenate((data_file1[b'data'], data_file2[b'data'], data_file3[b'data'], data_file3[b'data'], data_file5[b'data']))
    training_labels = np.concatenate((data_file1[b'labels'], data_file2[b'labels'], data_file3[b'labels'], data_file4[b'labels'], data_file5[b'labels']))
    train_y = np.zeros((len(training_labels), 10))
    for i in range((len(training_labels))):
        train_y[i][training_labels[i]] = 1

    test_x = test_file[b'data']
    testing_labels = test_file[b'labels']
    test_y = np.zeros((len(testing_labels), 10))

    for i in range((len(testing_labels))):
        test_y[i][testing_labels[i]] = 1
    # train_x = data_file['train_x']
    # train_y = data_file['train_y']
    # test_x = data_file['test_x']
    # test_y = data_file['test_y']

    print("train_x, train_y, test_x, test_y",train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        
    train_x, val_x, train_t, val_t = train_test_split(train_x, train_y, test_size=0.2)

    train_x = normalize_data(train_x)
    val_x = normalize_data(val_x)
    test_x = normalize_data(test_x)   
    
    
    
    return train_x.T, train_t.T, val_x.T, val_t.T, test_x.T, test_y.T   

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict