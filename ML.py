#!/usr/bin/env python
# coding: utf-8

# #### Preparing and PProcessing Data for Modelling

# In[1]:


# import required libraries
import pandas as pd 
from pandas import Series
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# importing predictive classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

### importing metrics for MODEL EVALUATION
from sklearn.metrics import (classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import warnings # mute warnings

print("import libraries are loaded")


# In[2]:


#reading in data
df_train = pd.read_csv("train (1).csv")


# In[3]:


df_train.head(2)


# In[4]:


df_train.info()


# In[5]:


#descriptive statistics of the numerical features
df_train.describe()


# In[6]:


#statistical describtion of the categorical features
df_train.describe(exclude='int64').T


# In[7]:


# count the number of unique values to identify redundant features
{x: len(df_train[x].unique()) for x in  df_train.columns}


# Observations
# 
# Features EmployeeCount,Over18 and StandardHours (only 1 unique value: will add no new or relevant information to our model)
# EmployeeNumber (max number of 1058 unique values): serves no purpose to our analysis
# moving further, these feature will be dropped

# #### Exploratory Data Analysis

# #### Bivariate Analysis

# In[8]:


# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Univariate Analysis: Histogram of Age and Bar Chart for Attrition
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_train['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Age')

plt.subplot(1, 2, 2)
sns.countplot(x='Attrition', data=df_train)
plt.title('Attrition Counts')

plt.tight_layout()
plt.show()


# In[9]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Department', data=df_train)
plt.title('Count of Employees by Department')

plt.subplot(1, 2, 2)
sns.countplot(x='EducationField',data=df_train)
plt.title('Count of Employees by Education Field')


# In[ ]:





# Distribution of Age: The histogram shows the frequency distribution of ages among the employees, with a kernel density estimate overlay to show the distribution curve.
# Attrition Counts: The bar chart displays the counts of employees who have left the company (1) and those who haven't (0).

# Count by Department: The bar chart illustrates the count of employees distributed across different departments. Research & Development has the highest number of employees compared to other departments.
#     
# Count by Education Field: The bar chart for education fields shows that 'Life Sciences' and 'Medical' are the most common fields of study among the employees.

# In[10]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Age', y='TotalWorkingYears', data=df_train, hue='Attrition', palette='coolwarm', alpha=0.6)
plt.title('Age vs. Total Working Years')

plt.subplot(1, 2, 2)
sns.scatterplot(x='DailyRate', y='YearsAtCompany', data=df_train, hue='Attrition', palette='coolwarm', alpha=0.6)
plt.title('Daily Rate vs. Years At Company')

plt.tight_layout()
plt.show()


# Age vs. Total Working Years: The scatter plot reveals the relationship between an employee's age and their total working years. As expected, older employees generally have more working years, and this plot also colors points by attrition status, giving insights into which group is more likely to leave.
# 
# Daily Rate vs. Years at Company: This scatter plot shows how daily rate correlates with the number of years an employee has been with the company, also colored by attrition status.

# In[11]:


# Create a figure to host the plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Bivariate Analysis: 'Age' vs 'Attrition'
sns.boxplot(x='Attrition', y='Age', data=df_train, ax=ax[0])
ax[0].set_title('Attrition vs Age')
ax[0].set_xlabel('Attrition (0 = No, 1 = Yes)')
ax[0].set_ylabel('Age')

# Bivariate Analysis: 'TotalWorkingYears' vs 'Attrition'
sns.boxplot(x='Attrition', y='TotalWorkingYears', data=df_train, ax=ax[1])
ax[1].set_title('Attrition vs Total Working Years')
ax[1].set_xlabel('Attrition (0 = No, 1 = Yes)')
ax[1].set_ylabel('Total Working Years')

# Display the plots
plt.tight_layout()
plt.show()


# Attrition vs Age: The boxplot shows that younger employees tend to have higher attrition rates compared to older employees. Employees who stay tend to be older on average.
# 
# Attrition vs Total Working Years: Similar to age, employees with fewer total working years appear to have higher attrition rates. Those with longer tenure are more likely to stay.

# #### Multivariate analysis

# In[12]:


# Selecting a subset of columns for multivariate analysis
cols = ['Age', 'TotalWorkingYears', 'YearsAtCompany', 'DailyRate', 'Attrition']
sns.pairplot(df_train[cols], hue='Attrition', palette='coolwarm')
plt.suptitle('Pair Plot of Selected Variables Colored by Attrition', y=1.02)
plt.show()


# In[13]:


# Selecting a subset of variables for correlation analysis
selected_columns = ['Age', 'TotalWorkingYears', 'YearsAtCompany', 'DailyRate', 'DistanceFromHome']
subset_data = df_train[selected_columns]

# Calculating correlation matrix
correlation_matrix = subset_data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Selected Variables')
plt.show()



# Age and Total Working Years: There's a strong positive correlation between age and total working years, which is expected as more years in life typically mean more years working.
# 
# Age and Years at Company: Age also has a moderate positive correlation with the number of years at the company, suggesting that older employees tend to have longer tenures at the company.
# 
# Total Working Years and Years at Company: This shows a strong correlation, indicating that employees who have worked more years overall also tend to have spent a significant portion of those years at this particular company.
# 
# Daily Rate and Distance from Home: These variables show very little to no correlation with other variables, indicating that they do not significantly influence or are influenced by other factors like age or years worked.
# 

# ##### what are the distributions of our target features(Attrition), and (Monthly income)?

# In[14]:


# Create a figure to host the plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Distribution of 'Attrition'
sns.countplot(x='Attrition', data=df_train, ax=ax[0])
ax[0].set_title('Distribution of Attrition')
ax[0].set_xlabel('Attrition (0 = No, 1 = Yes)')
ax[0].set_ylabel('Count')

# Distribution of 'MonthlyIncome'
sns.histplot(df_train['MonthlyIncome'], bins=30, kde=True, ax=ax[1])
ax[1].set_title('Distribution of Monthly Income')
ax[1].set_xlabel('Monthly Income')
ax[1].set_ylabel('Frequency')

# Display the plots
plt.tight_layout()
plt.show()


# Distribution of Attrition: The bar chart shows the counts of employees who have left (1) versus those who haven't (0). It appears that a larger proportion of the dataset consists of employees who have not left the company.
#     
# Distribution of Monthly Income: The histogram of 'Monthly Income' shows a right-skewed distribution, indicating that most employees have lower to mid-range salaries, with fewer employees earning higher salaries.

# ##### explore the relationships between the target features with other predictor features

# In[15]:


# Create a figure to host the plots
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Attrition vs Age
sns.boxplot(x='Attrition', y='Age', data=df_train, ax=ax[0, 0])
ax[0, 0].set_title('Attrition vs Age')
ax[0, 0].set_xlabel('Attrition')
ax[0, 0].set_ylabel('Age')

# Attrition vs Job Role
sns.countplot(x='JobRole', hue='Attrition', data=df_train, ax=ax[0, 1])
ax[0, 1].set_title('Attrition by Job Role')
ax[0, 1].set_xlabel('Job Role')
ax[0, 1].set_ylabel('Count')
ax[0, 1].tick_params(axis='x', rotation=45)

# Monthly Income vs Education
sns.boxplot(x='Education', y='MonthlyIncome', data=df_train, ax=ax[1, 0])
ax[1, 0].set_title('Monthly Income by Education Level')
ax[1, 0].set_xlabel('Education Level')
ax[1, 0].set_ylabel('Monthly Income')

# Monthly Income vs Job Role
sns.boxplot(x='JobRole', y='MonthlyIncome', data=df_train, ax=ax[1, 1])
ax[1, 1].set_title('Monthly Income by Job Role')
ax[1, 1].set_xlabel('Job Role')
ax[1, 1].set_ylabel('Monthly Income')
ax[1, 1].tick_params(axis='x', rotation=45)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# Attrition vs Age: The boxplot shows a trend where younger employees are more likely to leave the organization compared to older ones. Lower age groups have higher attrition rates.
#     
# Attrition vs Job Role: The count plot illustrates that certain job roles might have higher attrition rates. The visibility of which roles are more prone to turnover can help tailor specific retention strategies.
#     
# Monthly Income vs Education: The boxplot indicates that higher education levels tend to be associated with higher monthly incomes, although the spread and outliers suggest there's variability within each education level.
#     
# Monthly Income vs Job Role: Different job roles command significantly different income levels, with roles like managers and executives typically at the higher end of the income scale, while roles like sales representatives are at the lower end.

# ##### Data Preprocessing

# In[16]:


#dropping off some redundant features
df_train.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1,inplace=True)
#creating a copy of the dataset for part B
df2_train = df_train.copy()


# In[17]:


#encode the categorical features to numerical ones
df_train = pd.get_dummies(df_train,drop_first=True)


# In[18]:


#segment dataset into data and target label
y = df_train.pop('Attrition')


# In[19]:


# scale dataset features
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#identifying key featurs from the data set
#plotting a feature importance chart
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_train,y),columns=df_train.columns)

#weill use a random classifier model to identify the importance features
model = RandomForestClassifier()
# fit the model
model.fit(df_scaled, y)
feature_names = list(df_train.columns)
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 7))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='lightgreen', align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[20]:


# split the DataFrame into train and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_scaled, y, train_size=0.7,random_state=1)


# In[21]:


# BUILDING A BASE MODEL
log_clf = LogisticRegression()
log_clf.fit(x_train,y_train)


# In[22]:


#creating a prediction file
log_pred = log_clf.predict(x_test)

#model accuracy score
print("Logistic Regression score on test data: ",format(log_clf.score(x_test,y_test)))


# In[23]:


#helpful function for visualizing confusion matrix
def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# In[24]:


confusion_matrix_sklearn(log_clf,x_test,y_test)


# In[25]:


print(classification_report(y_test,log_pred))


# In[ ]:





# In[26]:


# BUILDING A BASE MODEL
#fitting Decision tree into the training set
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(x_train,y_train)


# In[27]:


#creating a prediction file
#predicting the test set results
y_pred= DT_classifier.predict(x_test)

#model accuracy score
print("Decision Tree score on test data: ",format(DT_classifier.score(x_test,y_test)))


# In[28]:


confusion_matrix_sklearn(DT_classifier,x_test,y_test)


# In[29]:


print(classification_report(y_test,y_pred))


# #### Hyperparameter tuning

# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Set the parameters by cross-validation
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Create a Decision Tree Classifier
dt = DecisionTreeClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[31]:


# Instantiate the classifier with the best parameters
best_dt = grid_search.best_estimator_

# Predict on the test data
y_pred = best_dt.predict(x_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy on test data: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #### Regression Analysis

# Preparing and Processing Data for Modelling
# data preprocessing, variable encoding, data scaling and normalization

# In[32]:


# importing predictive regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[33]:


#segmenting target label from the dataset
target = df2_train.pop('MonthlyIncome')
df2_train.drop('Attrition',axis=1,inplace=True)


# In[34]:


#encode the categorical features to numerical ones
df2_train = pd.get_dummies(df2_train,drop_first=True)


# In[35]:


#plotting a feature importance chart
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df2_train,target)

model = RandomForestClassifier()
# fit the model
model.fit(df_scaled, target)
feature_names = list(df_train.columns)
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 7))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='lightgreen', align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[36]:


# scale dataset features
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df2_train),columns=df2_train.columns)


# Splitting data into training and evaluation datasets
# Implementing Machine Learning/model Building and Training
# creating a predictions file

# In[37]:


# split the DataFrame into train and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.3,random_state=1)


# In[38]:


# Fit a linear regression model on the training set
lreg_model = LinearRegression()
lreg_model.fit(x_train, y_train)


# In[39]:


#creating a prediction file
lreg_pred = lreg_model.predict(x_test)


# we will compute the model's r_squared score (r2_score) which is a measure of how "good afit" the linear model is for modelling this kind of data
# here we will use business metrics such as the mean squared error(mse) and the root mean squared error (rsme) model Evaluation to evaluate the linear regression's model's performance
# we'ill also visualize the plot of the model's predicted label vs the actual label

# In[40]:


# Evaluate the model using the test data
mse = mean_squared_error(y_test, lreg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, lreg_pred)

print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)


# In[41]:


# Plot predicted vs actual
plt.scatter(y_test, lreg_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('monthly salaries Predictions')
# overlay the regression line
z = np.polyfit(y_test, lreg_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()


# In[42]:


#visualizing model coefficients 
predictors = x_train.columns
coef = Series(lreg_model.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')


# #### Phase 3 Hyper Parameter 0ptimization
# - hyperparameter optimization is the art and science of improving our model's performances
# - we will be  implementing and optimizing a GradientBoostingRegressor model

# In[43]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
# Use a Gradient Boosting algorithm
alg = GradientBoostingRegressor()

# Try these hyperparameter values
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

# Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(x_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")


# measuring performance on test set
print ("Applying best model on test data:")
best_mod = gridsearch.best_estimator_
pred = best_mod.predict(x_test)


# In[44]:


# Evaluate the model using the test data
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)


# In[45]:


# Plot predicted vs actual
plt.scatter(y_test, pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('monthly salaries Predictions')
# overlay the regression line
z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()


# Observations we can see an improved performance through the optimized gradient boosting model, with an r squared score of 0.96, which is better than the linear regression's model. we will sumbit this as our best performing model and productionize it

# In[ ]:





# #### Productionizing Our Model
# this is the last stage of the machine learning pipeline, and the main aim here points to how the users use/consume the model.
# there are alot of ways an ML Model can be used
# - it can be embedded into an application to be used by users online via an API on  web interfaces or on mobile devices
# - It can be used to create reports or dashboards that will be used by the organisation in making key business decisions
# - it can be consumed via streaming or batch methods
# 
# **In this case scenario, we will simulate the use of the model on a new dataset and use it to make relevant predictions**

# In[46]:


testing = pd.read_csv("test.csv")
#testing.info()


# In[47]:


#Productionizing the best performing model
# Serializing the best model for subsequent and easy usage
import joblib

# Save the model as a pickle file
filename = './optimized_mod.pkl'
joblib.dump(best_mod, filename)


# ** Now to stimulate a typical production environment, we will use the test data set( the other data set in the data folder) as the new employeedata set to 
# predict 

# In[48]:


#apply transforms to the new data similar to the training dataset
testing.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours','MonthlyIncome'], axis=1,inplace=True)
testing = pd.get_dummies(testing,drop_first=True)

prediction = best_mod.predict(testing)
testing['pred'] = prediction

print(testing.head())


# In[ ]:




