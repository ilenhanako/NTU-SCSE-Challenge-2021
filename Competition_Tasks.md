## Task:
 ### Preliminary Round: placed 5/27
 The top 10 teams were determined from the preliminary round private leader board at the end of the round

- using deep learning: evaluate math expressions presented in the form of images. The images are black and white and are of dimensions (55, 135). 
- The numbers are single-digit numbers from 0 to 9, while the operator can be anything from addition, subtraction, multiplication and division

### Final Round: BabyMath
- Applied the model to a web

## Evaluation Metrics: Accuracy
- using exact string matching
- (number of correct matches)/(total number of rows)

## Datasets:
originally given: 
- Train (train.csv): 100,000 images
- Test (test.csv): 20,000 images

NOTE: 
added external datasets (MNIST data). only train datasets is attached in this folder.

## Data Fields:
- __id__ - filename of the image, whose corresponding - image can be found in the image folder
- __answer__ - the evaluated expression in the image, rounded and reported to 2 decimal places
- __num1__ - the first number
- __op__ - the operator
- __num2__ - the second number