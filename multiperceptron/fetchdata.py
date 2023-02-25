# %%
# taken and adapted from https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
import numpy as np
import requests, gzip, os, hashlib

path='./data'

def sample(x_data, y_data, num, predictable=True):
    if predictable: np.random.seed(0)
    else: np.random.seed()
    indexes = np.random.randint(0, x_data.shape[0], size=num)
    x = x_data[indexes].reshape((-1, 28 * 28))
    y = y_data[indexes]
    return x, y

def show_char(array): # display an ASCII version of a loaded character
    array = array.reshape((28, 28))
    print(np.array2string(array, formatter={'int':lambda x: '*' if x > 0 else ' '}))

def save_model(log=True):
    fp = os.path.join(path, 'trained')
    with open(fp, "wb") as f:
        np.savez_compressed(f, l1, l2)
    if log: print('Model saved.')

def load_model(log=True):
    fp = os.path.join(path, 'trained')
    with open(fp, "rb") as f:
        npzfile = np.load(f)
        l1 = npzfile['arr_0']
        l2 = npzfile['arr_1']
    if log: print('Model loaded.')
    return l1, l2

def fetch(url:str, log): # get data
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        if log: print('Reading local data...')
        with open(fp, "rb") as f:
            data = f.read()
    else:
        if log: print('Retrieving data from the web...')
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

def get_training_data(log=True): # get the training data
    X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", log)[0x10:].reshape((-1, 28, 28))
    Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", log)[8:]
    return X, Y

def get_test_data(log=True): # get the test data
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", log)[0x10:].reshape((-1, 28 * 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", log)[8:]
    return X_test, Y_test

def init_connections(x:int, y:int, predictable=True): # create and randomly init the connections between two layers
    if predictable: np.random.seed(0)
    else: np.random.seed()
    layer = np.random.uniform(-1., 1., size=(x, y)) / np.sqrt(x * y)
    return layer.astype(np.float32)

def sigmoid(x): # sigmoid function
    return 1 / (np.exp(-x) + 1)    

def d_sigmoid(x): # derivative of sigmoid
    return (np.exp(-x)) / ((np.exp(-x) + 1)**2)

def softmax(x): # softmax function
    y = np.exp(x - x.max())
    return y / np.sum(y,axis=0)

def d_softmax(x): # derivative of softmax
    y = np.exp(x - x.max())
    return y / np.sum(y,axis=0) * (1 - y / np.sum(y,axis=0))

def forward_backward_pass(l1, l2, x, y): # forward and backward pass for training
    targets = np.zeros((len(y),10), np.float32) # labels as binary vectors (1 at the right value, 0 everywhere else)
    targets[range(targets.shape[0]), y] = 1
    x_l1 = x.dot(l1) # forward pass
    x_sigmoid = sigmoid(x_l1)
    x_l2 = x_sigmoid.dot(l2)
    out = softmax(x_l2)
    error = 2 * (out - targets) / out.shape[0] * d_softmax(x_l2)
    update_l2 = x_sigmoid.T @ error # (rem: x.T is trasposed x; x @ y is matrix multiplication of x and y)
    error = (l2.dot(error.T)).T * d_sigmoid(x_l1)
    update_l1 = x.T @ error
    return out, update_l1, update_l2

def train(layer1, layer2, data, labels, epochs=4000, lr=0.001, batch=128, predicatable=True, log=True):
    if log: print(f'\n*** Training with {epochs} epochs with {batch} batches each ***\nEpoch\tAccuracy')
    data_val = data.reshape((-1, 28 * 28))
    for i in range(epochs):
        x, y = sample(data, labels, batch, predicatable)
        _, update_l1, update_l2 = forward_backward_pass(layer1, layer2, x, y)
        layer1 = layer1 - lr * update_l1
        layer2 = layer2 - lr * update_l2
        if log and i % 500 == 0:
            classified = classify(layer1, layer2, data_val)
            print(f'{i}\t{accuracy(classified, labels):.3f}')
    return layer1, layer2

def classify(l1, l2, x): # use the trained net to classify 
    x_l1 = sigmoid(x.dot(l1))
    x_l2 = x_l1.dot(l2)
    return np.argmax(softmax(x_l2),axis=1)

def accuracy(data, ref):
    return (data == ref).mean()

# *************************************************************

train_it = False
save_it = False
load_it = True
test_it = True
l1, l2 = '', ''

# *************************************************************

if train_it:
    # fetch data
    X, Y = get_training_data(log=False)
    # create and randomly init layers: 0, input: 28*28; 1, hidden: 128; 2, output: 10
    l1 = init_connections(28 * 28, 128, predictable=False)
    l2 = init_connections(128, 10, predictable=False)
    l1, l2 = train(l1, l2, X, Y, epochs=10000, predicatable=False)

if save_it:
    save_model()

if load_it:
    l1, l2 = load_model()

if test_it:
    X_test, Y_test = get_test_data(log=False)
    num = 20
    print(f'\n*** Testing {num} samples ***\nIndex\tWrong\tCorrect')
    x, y = sample(X_test, Y_test, num=num, predictable=False)
    res = classify(l1, l2, x)
    for i in range(num):
        if res[i] != y[i]: print(f'{i}\t{res[i]}\t{y[i]}')
    print(f'\nAccuracy: {(res == y).mean()}')
    # and, given the index i, show_char(x[i]) displays the shape of char y[i]


