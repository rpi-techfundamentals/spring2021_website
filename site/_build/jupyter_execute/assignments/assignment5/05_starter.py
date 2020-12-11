## Homework 05 - Instructions

![](https://github.com/rpi-techfundamentals/hm-01-starter/blob/master/notsaved.png?raw=1)

**WARNING!!!  If you see this icon on the top of your COLAB sesssion, your work is not saved automatically.**


**When you are working on homeworks, make sure that you save often. You may find it easier to save intermident copies in Google drive. If you save your working file in Google drive all changes will be saved as you work. MAKE SURE that your final version is saved to GitHub.** 

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel → Restart) and then run all cells (in the menubar, select Cell → Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment graded from answers to questions and automated tests that should be run at the bottom. Be sure to clearly label all of your answers and commit final tests at the end.  

files = "https://github.com/rpi-techfundamentals/introml_website_fall_2020/raw/master/files/assignment5.zip" 
!pip install otter-grader && wget $files && unzip -o assignment5.zip

#Run this. It initiates autograding. 
import otter
grader = otter.Notebook()

### Load Data
We have our titanic dataset that is a bit different from what we have had previously. Load the train-new.csv and test-new.csv into dataframes train and test.


# Load the data here
import pandas as pd
#Load the train-new.csv and test-new.csv into dataframes train and test
train = pd.read_csv('train-new.csv')
test = pd.read_csv('test-new.csv')
train.head()

## Question 1
(1) Investigate the data a little bit. What is different from some of the titanic datasets we have used in the past? (For example, compare against the data in the Kaggle Baseline notebook). 

man1="""

"""

# Generating Dummy Variables
Before we do analysis of the titanic dataset, we have to select out our features, for the train and the test set, which we shall label `X_train`, and `X_test`.

As a part of this we need to generate `n-1` dummy variables for each one of our categorical columns.  The resulting dataframes should be all numeric and have all of these columns below (in the correct order).

Follow the example above to generate a new value for `X_train` and `X_test` utilizing all the data.
```
['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']

```
*Hint, try: 
`help(pd.get_dummies)`* 

You should also set `y` to the Survived column.

X = pd.get_dummies(train, drop_first=True, columns=['Pclass','Sex','Cabin','Embarked'])

X.columns.values

#Answer Here 
X = pd.get_dummies(train, drop_first=True, columns=['Pclass','Sex','Cabin','Embarked'])
X_train=X.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']]
X_train=X[['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']]
X_test=X.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']]
X_test=X[['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']]
y=train['Survived']



X_train.columns.values.toset()

grader.check('q01')

## Split Training Set For Cross Validation
(2.) We want to split up our training set `X` so that we can do some cross validation. Specifically, we will start to use the term validation set for the set we will use to validate our model. 

In doing so below, use the sklearn methods to do a train test (i.e., validation) split.  

From X y dataframe, generate the following dataframes by drawing the data **randomly**  from the train dataframe 80% of the data in train and 20% of the data in test.  So that you get repeatable results, set the `random_state=100`. This will set a "seed" so that your random selection will be the same as mine and you will pass the internal tests. 

train_X, val_X, train_y, val_y


#Answer Here
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, 
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=100)
print(int(train_y.sum()),int(val_y.sum()))
print(int(train_X.Cabin_H.sum()),int(val_X.Cabin_H.sum()))

grader.check('q02')

### Perform Nearest Neighbor Classification (KNeighborsClassifier)
(3.) Using the default options (i.e., all default hyperparameters), perform nearest neighbor classification. Calculate the accuracy measure using `metrics.accuracy_score`.

Train your model using the training data and access the accuracy of both the training and validation data.

*Note: You only train the model once...on the training data. You then assess the performance on both the training and validation data.*

Assign the following variables:

`knn0_train_y` = The KNN prediction for the `train_X` data. 

`knn0_val_y`   = The KNN prediction for the `val_X` data. 

`knn0_train_accuracy` = The accuracy for the `knn0_train_y` prediction. 

`knn0_val_accuracy` = The accuracy for the `knn0_val_y` prediction. 


#Answer Here 

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#This creates a model object.
classifier = KNeighborsClassifier()
#This fits the model object to the data.
classifier.fit(train_X, train_y)
#This creates the prediction. 
knn0_train_y = classifier.predict(train_X)
knn0_val_y = classifier.predict(val_X)
knn0_train_accuracy=metrics.accuracy_score(train_y, knn0_train_y) 
knn0_val_accuracy = metrics.accuracy_score(val_y, knn0_val_y)
print(int(knn0_train_y.sum()), int(knn0_val_y.sum()),round(knn0_train_accuracy,2), round(knn0_val_accuracy,2))

grader.check('q03')

### Confusion Matrix
We can utilize a confusion matrix to be able to understand misclassifications a bit more. This will give us a full idea of the true positives, true negatives, false positives, and false negatives.  

See the documentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html). 

You can utilize the syntax below to generate knn_mat1_train and knn_mat1_test.  
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
```
**(4.)  Explain what each of the four quadrants of the confusion matrix means. **


#Answer here
man4= """

"""

### Create Confusion Matrix for the Training and Validation Predictions
(5) Create a confusion matrix for each of the training and valiation predictions. 
`knn0_con_train` A confusion matrix for the training data.  
`knn0_con_val` A confusion matrix for the validation data. 

#Answers 
from sklearn.metrics import confusion_matrix
knn0_con_train=confusion_matrix(train_y, knn0_train_y)
knn0_con_val=confusion_matrix(val_y, knn0_val_y)

print(knn0_con_train[0,0], knn0_con_train[0,1])
print(knn0_con_val[1,0], knn0_con_val[1,1])

grader.check('q05')

## Hyperparameter Tuning

(6) You created a single model using the default parameters.  However, we want to adjust the parameters to try and improve the model.  

Examine the documentation on KNN and see some of the different parameters that you can adjust. 

[Scikit Learn Documentation](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning).

Assign the following variables:

`knn1_train_y` = The KNN prediction for the `train_X` datafor your improved model. 

`knn1_val_y`   = The KNN prediction for the `val_X` data for your improved model. 

`knn1_train_accuracy` = The accuracy for the `knn1_train_y` prediction. 

`knn1_val_accuracy` = The accuracy for the `knn1_val_y` prediction. 


#Answers
classifier2 = KNeighborsClassifier(n_neighbors=3)
#This fits the model object to the data.
classifier2.fit(train_X, train_y)
#This creates the prediction. 
knn1_train_y = classifier2.predict(train_X)
knn1_val_y = classifier2.predict(val_X)
knn1_train_accuracy=metrics.accuracy_score(train_y, knn1_train_y) 
knn1_val_accuracy = metrics.accuracy_score(val_y, knn1_val_y)
print(int(knn1_train_y.sum()), int(knn1_val_y.sum()),round(knn1_train_accuracy,2), round(knn1_val_accuracy,2))

grader.check('q06')

### Other Models

(7.) Test Logistic regression and 1 other algorithms/models (your choice).  Provide a summary of the best performance below. 

Use any of the available classification models. You should show and comment code

[Scikit Learn Documentation](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning).

*Make sure your clearly indicate the accuracy of the Logistic regression model your other model. Assess which model worked best considering all your efforts.*
  

#Answer here
man7= """


"""

#This runs all tests. 
grader.check_all()

