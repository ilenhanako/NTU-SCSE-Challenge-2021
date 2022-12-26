# Import Library
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from keras import models, layers
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout

from PIL import Image
from IPython.display import Image as IPythonImage, display
from keras.optimizers import Adam
from keras import backend as K

# External datasets: MNIST
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.utils import np_utils

# Load CSV Datasets
IMG_PATH = "/kaggle/input/images-dataset/images/"
INPUT_PATH = "/kaggle/input/"
train = pd.read_csv(INPUT_PATH + 'train-dataset/train.csv')
test = pd.read_csv(INPUT_PATH + 'test-dataset/test (1).csv')

#MNIST_TRAIN_SIZE , max value is 60000
MNIST_TRAIN_SIZE = 2000
BATCH_SIZE = 128

#DO NOT FORGET TO ADJUST THE train_size (max 100000) and valid_size (?) somewhere down below

# Checking datasets sizes
(Mx_train, My_train), (Mx_test, My_test) = mnist.load_data()
print(Mx_train.size)
print(My_train.size)
print(Mx_test.size)
print(My_test.size)

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(Mx_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(My_train[i]))
  plt.xticks([])
  plt.yticks([])
fig

train.head()

'''
To display the data:

# img = mpimg.imread('../input/images-dataset/images/TRAIN00000.jpg')
'''
img = mpimg.imread(IMG_PATH + 'TRAIN00000.jpg')
plt.figure(figsize=(10,10))
plt.imshow(img, cmap = 'gray')

# Preprocess CSV Data
X = train['id']

#Y
def wow(operator):
    if operator == "minus":
        return("-")
    elif operator == "plus":
        return("+")
    elif operator == "divide":
        return("/")
    else:
        return("*")
 
train['temp'] = train.apply(lambda i: wow(i['op']), axis = 1)
train['Y']=train.apply(lambda z: (str(z['num1'])+ str(z['temp']) + str(z['num2'])), axis=1)
Y = train['Y']
train.head()

# Load Image
def preprocess(img):
    (h, w) = img.shape
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


train_x = []

train_size = 10000
for i in range(train_size):
    img_dir = IMG_PATH +train.loc[i, 'id']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    train_x.append(image)

print(len(train_x))

# Add MNIST Mx_train to train_x
for i in range(MNIST_TRAIN_SIZE):
    image = preprocess(Mx_train[i])
    image = image/255.
    train_x.append(image)

print(len(train_x))

valid_x = []
valid_size = 500

for i in range(valid_size):
    img_dir = IMG_PATH +train.loc[i+80000, 'id']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    valid_x.append(image)
train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
print(valid_x.shape)
print(train_x.shape)

#Preparing Label for CTC Loss
characters = u"0123456789+-/* "
max_str_len = 3 # max length of input labels
num_of_characters = len(characters) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 8 # max length of predicted labels

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(characters.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=characters[ch]
    return ret
name = '2+2'
print(name, '\n',label_to_num(name))

'''
- train_y contains the true labels converted to numbers and padded with -1. The length of each label is equal to max_str_len.

- train_label_len contains the length of each true label (without padding)

- train_input_len contains the length of each predicted label. The length of all the predicted labels is constant i.e number of timestamps - 2.

- train_output is a dummy output for ctc loss.
'''
train_y = np.ones([train_size + MNIST_TRAIN_SIZE, max_str_len]) * -1
train_label_len = np.zeros([train_size + MNIST_TRAIN_SIZE, 1])
train_input_len = np.ones([train_size + MNIST_TRAIN_SIZE, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size + MNIST_TRAIN_SIZE])


for i in range(train_size):
    train_label_len[i] = len(train.loc[i, 'Y'])
    train_y[i, 0:len(train.loc[i, 'Y'])]= label_to_num(train.loc[i, 'Y']) 

# Add Mnist My_train to train_y
print(MNIST_TRAIN_SIZE)
print(train_size + MNIST_TRAIN_SIZE)
for i in range(MNIST_TRAIN_SIZE):
    # train_label_len[i] = len(train.loc[i, 'Y'])
    train_label_len[i+train_size] = 1
    
    # train_y[i, 0:len(train.loc[i, 'Y'])]= label_to_num(train.loc[i, 'Y']) 
    train_y[i+train_size, 0:1]= label_to_num("{}".format(My_train[i])) 
train_y

valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(train.loc[i+80000, 'Y'])
    valid_y[i, 0:len(train.loc[i+80000, 'Y'])]= label_to_num(train.loc[i+80000, 'Y'])    

print('True label : ',train.loc[100, 'Y'] , '\ntrain_y : ',train_y[100],'\ntrain_label_len : ',train_label_len[100], 
      '\ntrain_input_len : ', train_input_len[100])

# Building our model
train_x.shape
input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


# Train our Model
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.0001))
model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs=60, batch_size=BATCH_SIZE)


# Evaluate using Val Dataset
print(valid_size)
#TEST SUBMISSION
val_answer = []
val_loss = []
val_loss_index = []

plt.figure(figsize=(15, 10))

for i in range(valid_size):
    img_dir = IMG_PATH +train.loc[i+80000, 'id']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image2 = image
    image = preprocess(image)
    image = image/255.

    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    
    decoded = decoded[0, 0:8]
    val_answer.append(decoded)
    
    temp = val_answer[i]
    ans=num_to_label(temp)
    
    if ans != train.loc[i+80000, 'Y']:
        val_loss.append(ans)
        val_loss_index.append(i)
        
plt.subplots_adjust(wspace=0.2, hspace=-0.8)

print(val_loss)
print(int(val_loss_index[4]))



for i in range(4):    
#    i = 4 #please change yourself
    temp = int(val_loss_index[i])

    img_dir = IMG_PATH +train.loc[temp+80000, 'id']
    correct = train.loc[temp+80000, 'Y']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title((val_loss[i] + ' ' + correct), fontsize = 20)



# Predict on Test Dataset
test.head()
img_dir = IMG_PATH +test.loc[10, 'id']
image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
plt.imshow(image)

test
test_size = len(test)


#TEST SUBMISSION
answer = []

## TODO BEFORE SUBMIT, remove below line
test_size = 1000

for i in range(test_size):
    img_dir = IMG_PATH +test.loc[i, 'id']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.

    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    
    decoded = decoded[0, 0:10]
    answer.append(decoded)

submission = pd.read_csv('https://raw.githubusercontent.com/ilenhanako/adelaine/master/submission.csv')
submission.head()


# Evaluate Math Expressions
for i in range(test_size):
    temp = answer[i]
    
    # dynamic index of operator
    if int(temp[1]) >= 10 :
        num1_index = 0
        opt_index = 1
        num2_index = 2
    elif int(temp[2]) >= 10 :
        num1_index = 1
        
    num1 = int(temp[0])
    num2 = int(temp[2])
    oper = int(temp[1])
    
    if oper == 10: 
        ans = num1 + num2
    elif oper == 11:
        ans = num1 - num2
    elif oper == 13:
        ans = num1 * num2
    elif num2 == 0:
        ans = 0
    else:
        ans = num1 / num2
    
    x = "%.2f" % ans 
    submission.loc[i,'answer'] = x
    print(x)

compression_opts = dict(method = 'zip',
                       archive_name = 'submission.csv')
submission.to_csv('filename.zip', index=False, compression = compression_opts)

val_answer = []

for i in range(valid_size):
    img_dir = IMG_PATH +train.loc[i+80000, 'id']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.

    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    
    decoded = decoded[0, 0:10]
    val_answer.append(decoded)



# Garbage Loss Counter
garbageloss_counter = 0
val_loss_index = []
val_loss_answer = []

for i in range(valid_size):
    temp = val_answer[i]
    if int(temp[1]) > 9:
        ans = (temp[0:3])
    else:
        ans= (temp[1:4])
        garbageloss_counter = garbageloss_counter + 1
    
    ans = num_to_label(ans)
    
    key = train.loc[i+80000, 'Y']
    
    if ans != key:
        val_loss_index.append(i)
        val_loss_answer.append(ans)
print(garbageloss_counter)
n_loss = len(val_loss_index)
print(n_loss)

row = n_loss / 3

# Display Errors
plt.figure(figsize=(15, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    img_dir = IMG_PATH +train.loc[val_loss_index[i]+80000, 'id']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    title = str(train.loc[val_loss_index[i]+80000, 'Y']) + " " + str(val_loss_answer[i])
    plt.title(title, fontsize=12)
    plt.axis('off')

    
plt.subplots_adjust(wspace=0.2, hspace=0.2)
