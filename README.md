# MLAlgos
This contains my implementation of the Decision Tree Classifier. It uses only one hyperparamter - max_depth - to tune the model. I am looking to add more implementations of Machine Learning algorithms here.

## Usage - DecisionTreeClf
1. Download the mllib folder to your working directory.
2. Import the DecisionTreeClf in your python code and create an object of the same by passing the desired max-depth (100 is default).
3. Separate out the independent variables and dependent variable from your dataset and convert them to numpy arrays.
4. Use these arrays as your input to the fit() method of the DecisionTreeClf object.
5. Print the tree usinng printTreeInOrder() to see how the data is classified.
6. Use predict() to predict the class based on the inputs provided.
