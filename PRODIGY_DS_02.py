#!/usr/bin/env python
# coding: utf-8

# **Task-02
# 
# **Problem Statement: 
# 
# Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from Kaggle. Explore the relationships between variables and identify patterns and trends in the data.
# 
# Dataset used: Titanic dataset
# 
# **Description :
# 
# The Titanic dataset comprises information about passengers aboard the RMS Titanic during its maiden voyage in 1912. It contains 12 columns, including 'Survived' (0 for not survived, 1 for survived), 'Pclass' (passenger class), 'Name,' 'Sex,' 'Age,' 'SibSp' (siblings/spouses aboard), 'Parch' (parents/children aboard), 'Ticket' number, 'Fare' paid, 'Cabin' number, and 'Embarked' port. The dataset often contains 891 records. It serves as a classic dataset for data analysis and machine learning, used to explore factors affecting survival and develop predictive models. Variables include both numerical and categorical data. It's employed for tasks like visualization, feature engineering, and teaching data science concepts. Missing values are common, particularly in 'Age' and 'Cabin' columns, necessitating data cleaning.
# 
# Link: https://www.kaggle.com/datasets/brendan45774/test-file

# In[4]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Load the Titanic dataset and view the first 5 rows of the dataset
data = pd.read_csv("tested.csv")
data.head()


# **Performing basic EDA

# In[8]:


#Checking the dimension of the dataset:
print("The total rows in this dataset is:" ,data.shape[0] ,"\nThe total columns in this dataset is:" ,data.shape[1])


# In[10]:


#checking the type of data and missing values
data.info()


# **Data Cleaning

# In[12]:


# Check for missing values
missing_data = data.isnull().sum()
print("Missing Data:\n", missing_data)


# **Handling missing values
# 
# As the dtype of 'Age' and 'Fare' are Float so median is used for imputing the missing value
# 
# As the dtype of 'Cabin' is Object so mode is used for imputing the missing value

# In[14]:


# Handle missing values 
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Cabin'].fillna(data['Cabin'].mode(), inplace=True)


# In[16]:


# Check if there are still any missing values in the dataset
if data.isnull().any().any():
    print("There are missing values in the dataset.")
    print(data.isnull().sum())
else:
    print("There are no missing values in the dataset.")


# In[20]:


# Exploratory Data Analysis (EDA)
# Summary statistics
summary_stats = data.describe()
print("Summary Statistics:\n", summary_stats)


# **Performing Correlation to understand the important numerical features in the dataset:

# In[22]:


#Selecting the numerical columns:
print("The Numerical columns are: ")
data_numerical=data.select_dtypes(np.number)
data_numerical


# In[24]:


#  Correlation matrix (for continuous variables)
correlation_matrix = data_numerical.corr()
correlation_matrix


# Visualization
# 
# To explore the relationships between variables and identify patterns and trends in the data we will be performing the following visualizations:
# 
# Heatmap
# Pair Plots
# Box Plots
# Bar Plots
# Histograms
# 
# 1.Heatmap
# 
# A heatmap can show the correlations between numerical variables.

# In[26]:


# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Interpretation:
# 
# From the Diagonal Plot(Kernal Density Estimator): It represents the distribution of each variable independently.We can see the variable 'Age' is normally distributed and the variables 'Fare' , 'Parch' and 'SibSp' are right skewed.
# 
# Scatter Plots: The off-diagonal plots are scatter plots, and they show the relationship between pairs of variables. Here we see there is no such clear relationship between most of the variables.

# 2.Pair Plots (Scatter Matrix)
# 
# This plot helps visualize pairwise relationships between numerical variables.

# In[29]:


# Pairplot (for continuous variables)

sns.pairplot(data=data, vars=["Survived", "Pclass"	,"Age",	"SibSp",	"Parch",	"Fare"])


# Interpretation
# 
# Here, we can observe from the above heatmap that there is no such strong correlation among the numerical variables but the feature 'Pclass' and 'Fare' are moderately negetively correlated with a correlation coefficient of 0.58.
# 
# 

# 3.Box Plots
# 
# Box plots can reveal the distribution of a numerical variable within different categories.

# In[31]:


# Create subplots for 'Age' and 'Fare' by 'Pclass'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Box plot for 'Age' by 'Pclass'
sns.boxplot(x='Pclass', y='Age', data=data, hue='Survived', ax=axes[0])
axes[0].set_title('Age Distribution by Passenger Class and Survival')

# Box plot for 'Fare' by 'Pclass'
sns.boxplot(x='Pclass', y='Fare', data=data, hue='Survived', ax=axes[1])
axes[1].set_title('Fare Distribution by Passenger Class and Survival')

plt.show()


# Interpretation
# 1.Age Distribution by Passenger Class and Survival:
# 
# The left box plot shows the distribution of ages ('Age') for passengers in different passenger classes ('Pclass'). Each box represents a passenger class, and the boxes are divided by color (hue) to show the survival status ('Survived' - survived or not).
# 
# Key observations:
# 
# Passengers in the first class ('Pclass' 1) generally tend to be older than those in the second ('Pclass' 2) and third ('Pclass' 3) classes.
# Within each passenger class, the box plots are divided by color (survived or not). This division helps to see if there is a significant difference in age distribution between survivors and non-survivors.
# 
# 2.Fare Distribution by Passenger Class and Survival:
# 
# The right box plot shows the distribution of fares ('Fare') for passengers in different passenger classes ('Pclass'). Similar to the 'Age' plot, each box represents a passenger class, and the boxes are divided by color to show the survival status ('Survived').
# 
# Key observations:
# 
# Passengers in the first class ('Pclass' 1) generally paid higher fares than those in the second and third classes.
# Within each passenger class, the box plots are divided by color, allowing you to see differences in fare distribution between survivors and non-survivors.
# There is some variation in fare distribution, especially in the first and second classes, with some passengers in these classes paying significantly more.

# 4.Bar Plots
# Bar plots can show the distribution of categorical variables.

# In[33]:


# Create subplots for 'Pclass' and 'Sex' by 'Survived'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Bar plot for 'Pclass' by 'Survived'
sns.countplot(x='Pclass', data=data, hue='Survived', ax=axes[0])
axes[0].set_title('Survival by Passenger Class')

# Bar plot for 'Sex' by 'Survived'
sns.countplot(x='Sex', data=data, hue='Survived', ax=axes[1])
axes[1].set_title('Survival by Gender')

plt.show()


# Interpretation
# The bar plots illustrate the relationships between passenger class, gender, and survival on the Titanic. It clearly show that both passenger class and gender were significant factors influencing survival rates:
# 
# First-class passengers had a higher chance of survival compared to second and third-class passengers.
# Female passengers had a substantially higher survival rate than male passengers, reflecting the famous "women and children first" policy during the Titanic's evacuation.

# 5.Histograms
# Histograms help visualize the distribution of numerical variables.

# In[35]:


# Create subplots for 'Age' and 'Fare' by 'Survived'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Histogram for 'Age' by 'Survived'
sns.histplot(data=data, x='Age', hue='Survived', bins=20, kde=True, ax=axes[0])
axes[0].set_title('Age Distribution by Survival')

# Histogram for 'Fare' by 'Survived'
sns.histplot(data=data, x='Fare', hue='Survived', bins=20, kde=True, ax=axes[1])
axes[1].set_title('Fare Distribution by Survival')

plt.show()


# Interpretation
# These histograms provide insights into the distribution of passenger ages and fares, considering survival outcomes on the Titanic:
# 
# The age distribution suggests that children and older passengers had a higher chance of survival.
# The fare distribution indicates that passengers who paid higher fares had a better chance of survival.

# In[ ]:




