# To Load CIFAR dataset

Load CIFAR-10 Dataset. (https://www.cs.toronto.edu/~kriz/cifar.html)

Clone the repo

```
git clone https://github.com/KhurramAsadM004/NeuralNetwork.git
```

Open anaconda prompt and go into the directory. Run this command. 
```
python .\driver_nn.py --layer_dim 3072 100 50 50 20 10 --activations None relu relu relu relu softmax --optimizer adam --epochs 1000 --loss mce --batch_size 64 --early_stopping --patience 10 --convergence_threshold 1e-5 --mode train --data_dir ./Data/cifar/ --dataset=cifar
```