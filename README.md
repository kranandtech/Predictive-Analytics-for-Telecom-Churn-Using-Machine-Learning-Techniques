# Predictive-Analytics-for-Telecom-Churn-Using-Machine-Learning-Techniques

This project aims to predict customer churn in a telecom company using demographic and usage data. The workflow involves loading data, preprocessing, feature engineering, model training, and evaluation.

# Approach
* Import Required Libraries
  We begin by importing necessary libraries such as Pandas for data manipulation, Scikit-learn for preprocessing, model training, and evaluation.
* Load Data
  CSV file is loaded. Datset is from kaggle- https://www.kaggle.com/datasets/suraj520/telecom-churn-dataset

* Identify Churn Rate
We calculate the churn rate by determining the proportion of customers who have churned. This gives an initial understanding of the churn distribution in the dataset.

* Identify Categorical Variables
We inspect the data to identify categorical variables which need to be encoded before they can be used in machine learning models.

* One-Hot Encoding
Categorical variables such as telecom_partner, gender, state, city, and registration_event are transformed into a format suitable for machine learning models using One-Hot Encoding. This process converts categorical variables into a series of binary columns.

* Feature Scaling
We apply feature scaling to ensure that all numerical features have a standard scale. This is done using StandardScaler which standardizes features by removing the mean and scaling to unit variance.

* Define Features and Target
We separate the dataframe into features (input variables) and the target (output variable, which is churn). The customer_id column is dropped as it is not a feature.

* Split the Dataset
The data is split into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate the model's performance.

* Train Logistic Regression Model
A logistic regression model is instantiated and trained using the training data. Logistic regression is chosen for its simplicity and effectiveness in binary classification problems like churn prediction.

* Evaluate Logistic Regression Model
The performance of the logistic regression model is evaluated on the test set using metrics such as confusion matrix and classification report. These metrics help in understanding the model's accuracy, precision, recall, and F1-score.

* Train Random Forest Model
A random forest classifier is instantiated and trained using the training data. Random forests are powerful ensemble methods that can capture complex interactions between features.

*  Evaluate Random Forest Model
Similar to the logistic regression model, the random forest model is evaluated on the test set using confusion matrix and classification report. This helps in comparing the performance of both models.

# Conclusion - Random forest has highest accuracy 79% as comapred to Logistic Regression  which is 73%.

