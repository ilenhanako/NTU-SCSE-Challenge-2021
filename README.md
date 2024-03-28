# NTU SCSE Challenge 2021: 
NTU SCSE CHALLENGE [LINK](https://www.ntu.edu.sg/scse/news-events/news/detail/scse-computing-challenge-2021)
![Team CJPythons](https://www.ntu.edu.sg/images/librariesprovider118/news-events/scse-computing-challenge-202154f84e0e-2c74-4211-a795-816575dbf69f.jpg?Status=Master&sfvrsn=464ed816_3)
Awarded: Young Technopreneurs Award

## CNN and RNN
Our solution used [CNN and RNN](https://www.upgrad.com/blog/cnn-vs-rnn/) as our deep learning algorithm.

- CNN: used for image processing, recognition and classification.
- RNN: make better predictions by remembering vital details(eg: input received)

## MNIST Datasets from Keras
We incorporated additional datasets of handwritten digits: MNIST, [learn more about MNIST Datasets in Python](https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python)

## CTC Operation
To solve errors on duplicated characters. We need to calculate the loss value for the training samples to train the NN.
- learn more about [CTC](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c)

## Main preprocessing steps:

#### Loading and Checking Data:
CSV files containing training and testing datasets are loaded.
The MNIST dataset is also loaded and its size is printed.
The first few images of the MNIST dataset are displayed along with their labels.
The train.head() function is called, likely to check the first few rows of the training data.

#### Displaying an Image:
An image from the dataset is read and displayed using matplotlib.

#### Preprocessing CSV Data:
The 'id' column from the training dataset is used to locate images in the dataset.
A new column 'temp' is created in the dataframe to convert operation symbols to their string equivalents using a custom function.
Another column 'Y' is created to form a string representation of the mathematical expression combining the operands and the operator.

#### Loading and Preprocessing Images:
A preprocess function is defined to prepare the images. This function:
Initializes a blank white image of fixed size (64x256).
Crops the input image if it exceeds the fixed size.
Puts the cropped input image onto the blank white image to maintain a uniform size across all images.
Rotates the image 90 degrees clockwise to match the expected input orientation for the model.
This preprocessing function is applied to images from both the training and validation datasets, with specific subsets indicated by train_size and valid_size.
The processed images are added to lists train_x and valid_x.
MNIST images are also added to train_x after being processed by the same preprocess function.

#### Reshaping Images:
The lists of preprocessed images are converted to numpy arrays and reshaped to conform to the input shape expected by the neural network (with an added dimension for channels).

#### Preparing Labels for CTC Loss:
The code prepares the labels for use with Connectionist Temporal Classification (CTC) loss, which is commonly used for training sequence recognition models like the one being implemented.
Labels are converted from strings to numerical representations, with a mapping that assigns each character a unique integer based on the characters string.
The train_y array is initialized with -1, a placeholder for the CTC blank character.
train_label_len and train_input_len arrays are initialized to hold the lengths of the true labels and the predicted labels, respectively.
This preprocessing step is also done for the validation labels.

#### Image and Label Shape Confirmation:
The shapes of the processed image and label arrays are confirmed to ensure they are correctly structured for input into the model.
