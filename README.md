# Classification-model
## Chosen algorithms
I implemented four classification algorithms to find the best one that 
have high accuracy, which are:
<div style="text-align:justify">
<b>1. Logistic regression</b> <br>
This algorithm is used to classify the data, the used of this algorithm is to identify the relationship between a continuous dependent variable and one or more independent variables by represent a curved line between the data point, the line is like letter S and this is the reason of the name of this algorithm. Also, I didn’t 
change the hyperparameter default values.
**Accuracy:** 70.8 % **F1 score:** 70.1 %

**2. Support Vector Machine (SVM)** <br>
SVM is one of the most popular Supervised Learning algorithms, which is used for Classification and Regression problems. However, primarily, it is used for Classification problems in Machine Learning. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into 
classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a 
hyperplane. I change the default value of hyperparameter such as the kernel, by default is rbf, I replace it into linear. Also, gamma hyperparameter by default is scale, I change it into auto. 
**Accuracy:** 79.2 % **F1 score:** 78.9 %

**3. Naïve Bayes** <br>
This algorithm is used to classify the data. Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. So, it’s set of algorithms where all of them share a common principle. This algorithm is predicting membership probabilities for each class such as the probability that given data point belongs to a particular class. The class with the highest probability is considered as the most likely class. This is also known as Maximum A Posteriori (MAP). Also, I didn’t change the hyperparameter default values.
**Accuracy:** 76.1 % **F1 score:** 75.1 %

**4. Random forest classifier** <br>
Random forest is one of the most popular Supervised Learning algorithms, which is used for Classification as well asRegression problems. However, primarily, it is used for Classification problems in Machine Learning. Random forest is similar to decision tree, but the difference is that the decision tree is produce a single tree, but the random forest is builds multiple decision trees and merge them together to get a more accurate and stable prediction, and this why we called it forest.

The first step in this algorithm is to split the training data into several trees, each tree will train random data points from the origin training data, but each tree have different data points or records, this split of training data is called boosting sampling.

After that each tree was trained and it's ready to reception data to predict it, so each tree will give as the predicted class, then the final step is voting the result, meaning that the most frequent class will be chosen. In addition, we can choose the number of training tree in hyperparameter, one of them is n_estimators, and  the default value is 100, but I change it to 29 to increase the accuracy of the model, so it’s better to choose odd number of training tree (n_estimators) for if the equality of classes is occurring. Other hyperparameter I change it is the methodology of tree process which is criterion, by default it’s gini but I replace it into entropy (ID3), also to get same result every time I put the random_state = 42.
**Accuracy:** 86.5 % **F1 score:** 86.3 %

## Data processing
In our data set we have 16 features, and not all the feature is sensitive data to affect the accuracy for the model, so we ignore some feature in training model such as Nationality,PlaceofBirth, Semester, Relation, ParentAnsweringSurvey and ParentschoolSatisfaction was ignored is the training model so only 9 out of 16 was the train feature. 
Some of the column or feature have categorical value, so it’s not numerical value that can machine learning deal with, so we convert every categorical value into numerical value by preprocessing library in sklearn, in this library we have function called LavelEncoder, this function will use some technique to convert the data into numeric value. So we use this function for 5 out of 9 features to convert the values. Actually, instead of dropping the column that has categorical value, then adding a new converted column from the old column, we overwrite the new values on the old values in every specific column.
</div>
