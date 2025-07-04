# MNIST digit learning

A simple dense neural network using only numpy for use in introductory ML classes.
See https://sca.ksr.ch/doku.php?id=talit:neuronale_netze_kurs for the accompanying materials.

## Training
See [`neural_network.py`](neural_network.py): 

Train a new network from scratch:

```bash
python neural_network.py --dim 784 --dim 100 --dim 50 --dim 11 \
    --savefile mnist_best.npz \
    --datafile data_mnist/mnist_train.csv \
    --train --learningrate 0.02 --maxruns 400000
```

Retrain an existing network by first loading it:

```bash
python neural_network.py \
    --loadfile mnist_best.npz \
    --savefile mnist_best.npz \
    --datafile data_mnist/mnist_train.csv \
    --train --learningrate 0.01 --maxruns 200000
```

## Evaluate
```bash
python neural_network.py \
    --loadfile mnist_best.npz \
    --datafile data_mnist/mnist_test.csv \
    --maxruns 4000
```

Evaluate all models in folder:
```bash
python neural_network.py \
    --evalall \
    --datafile data_mnist/mnist_test.csv \
    --maxruns 4000
```

# Serve
See [`app.py`](app.py).

Serve a trained network and test with your own drawn digits. Note that the code expects to find the network in `mnist_best.npz`.

```bash
python -m flask run
```

Visit http://localhost:5000 to visit the app.

# Keras and Convolutional Networks

See [`keras.ipynb`](keras.ipynb) for an intro to convolutional neural networks (CNNs) and how to use them to improve digit recognition. 