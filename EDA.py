# Databricks notebook source
# MAGIC %md This notebook is copied from https://www.kaggle.com/code/ekami66/detailed-exploratory-data-analysis-with-python

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory data analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC This is the very first data analysis I do on my own. Please take the informations on this notebook with a grain of salt. I'm open to all improvements (even rewording), don't hesitate to leave me a comment or upvote if you found it useful. If I'm completely wrong somewhere or if my findings makes no sense don't hesitate to leave me a comment.
# MAGIC 
# MAGIC This work was influenced by some kernels of the same competition as well as the [Stanford: Statistical reasoning MOOC](https://lagunita.stanford.edu/courses/OLI/StatReasoning/Open/info)
# MAGIC 
# MAGIC The purpose of this EDA is to find insights which will serve us later in another notebook for Data cleaning/preparation/transformation which will ultimately be used into a machine learning algorithm.
# MAGIC We will proceed as follow:
# MAGIC 
# MAGIC <img src="http://sharpsightlabs.com/wp-content/uploads/2016/05/1_data-analysis-for-ML_how-we-use-dataAnalysis_2016-05-16.png" />
# MAGIC 
# MAGIC [Source](http://sharpsightlabs.com/blog/data-analysis-machine-learning-example-1/)
# MAGIC 
# MAGIC Where each steps (Data exploration, Data cleaning, Model building, Presenting results) will belongs to 1 notebook.
# MAGIC I will write down a lot of details in this notebook (even some which may seems obvious by nature), as a beginner it's important for me to do so.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC For the preparations lets first import the necessary libraries and load the files needed for our EDA

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

plt.style.use('bmh')

# COMMAND ----------

df = pd.read_csv('./data/home-prices.csv')
df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC From these informations we can already see that some features won't be relevant in our exploratory analysis as there are too much missing values (such as `Alley` and `PoolQC`). Plus there is so much features to analyse that it may be better to concentrate on the ones which can give us real insights. Let's just remove `Id` and the features with 30% or less `NaN` values.

# COMMAND ----------

# df.count() does not include NaN values
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
del df2['Id']
print("List of dropped columns:", end=" ")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")
print('\n')
df = df2

# COMMAND ----------

# MAGIC %md
# MAGIC <font color='chocolate'> Note: If we take the features we just removed and look at their description in the `data_description.txt` file we can deduct that these features may not be present on all houses (which explains the `NaN` values). In our next Data preparation/cleaning notebook we could tranform them into categorical dummy values.</font>

# COMMAND ----------

# MAGIC %md
# MAGIC Now lets take a look at how the housing price is distributed

# COMMAND ----------

print(df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

# COMMAND ----------

# MAGIC %md
# MAGIC <font color='chocolate'>With this information we can see that the prices are skewed right and some outliers lies above ~500,000. We will eventually want to get rid of the them to get a normal distribution of the independent variable (`SalePrice`) for machine learning.</font>

# COMMAND ----------

# MAGIC %md
# MAGIC Note: Apparently using the log function could also do the job but I have no experience with it

# COMMAND ----------

# MAGIC %md
# MAGIC ## Numerical data distribution

# COMMAND ----------

# MAGIC %md
# MAGIC For this part lets look at the distribution of all of the features by ploting them

# COMMAND ----------

# MAGIC %md
# MAGIC To do so lets first list all the types of our data from our dataset and take only the numerical ones:

# COMMAND ----------

list(set(df.dtypes.tolist()))

# COMMAND ----------

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now lets plot them all:

# COMMAND ----------

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations

# COMMAND ----------

# MAGIC %md
# MAGIC <font color='chocolate'>Features such as `1stFlrSF`, `TotalBsmtSF`, `LotFrontage`, `GrLiveArea`... seems to share a similar distribution to the one we have with `SalePrice`. Lets see if we can find new clues later.</font>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation

# COMMAND ----------

# MAGIC %md
# MAGIC Now we'll try to find which features are strongly correlated with `SalePrice`. We'll store them in a var called `golden_features_list`. We'll reuse our `df_num` dataset to do so.

# COMMAND ----------

df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))

# COMMAND ----------

# MAGIC %md
# MAGIC Perfect, we now have a list of strongly correlated values but this list is incomplete as we know that correlation is affected by outliers. So we could proceed as follow:
# MAGIC 
# MAGIC - Plot the numerical features and see which ones have very few or explainable outliers
# MAGIC - Remove the outliers from these features and see which one can have a good correlation without their outliers
# MAGIC     
# MAGIC Btw, correlation by itself does not always explain the relationship between data so ploting them could even lead us to new insights and in the same manner, check that our correlated values have a linear relationship to the `SalePrice`. 
# MAGIC 
# MAGIC For example, relationships such as curvilinear relationship cannot be guessed just by looking at the correlation value so lets take the features we excluded from our correlation table and plot them to see if they show some kind of pattern.

# COMMAND ----------

for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['SalePrice'])

# COMMAND ----------

# MAGIC %md
# MAGIC We can clearly identify some relationships. Most of them seems to have a linear relationship with the `SalePrice` and if we look closely at the data we can see that a lot of data points are located on `x = 0` which may indicate the absence of such feature in the house.
# MAGIC 
# MAGIC Take `OpenPorchSF`, I doubt that all houses have a porch (mine doesn't for instance but I don't lose hope that one day... yeah one day...).

# COMMAND ----------

# MAGIC %md
# MAGIC So now lets remove these `0` values and repeat the process of finding correlated values: 

# COMMAND ----------

import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 1): # -1 because the last column is SalePrice
    tmpDf = df_num[[df_num.columns[i], 'SalePrice']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))

# COMMAND ----------

# MAGIC %md
# MAGIC Very interesting! We found another strongly correlated value by cleaning up the data a bit. Now our `golden_features_list` var looks like this:

# COMMAND ----------

golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))

# COMMAND ----------

# MAGIC %md
# MAGIC <font color='chocolate'>We found strongly correlated predictors with `SalePrice`. Later with feature engineering we may add dummy values where value of a given feature > 0 would be 1 (precense of such feature) and 0 would be 0. 
# MAGIC <br />For `2ndFlrSF` for example, we could create a dummy value for its precense or non-precense and finally sum it up to `1stFlrSF`.</font>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC <font color='chocolate'>By looking at correlation between numerical values we discovered 11 features which have a strong relationship to a house price. Besides correlation we didn't find any notable pattern on the datas which are not correlated.</font>

# COMMAND ----------

# MAGIC %md
# MAGIC Notes: 
# MAGIC 
# MAGIC - There may be some patterns I wasn't able to identify due to my lack of expertise
# MAGIC - Some values such as `GarageCars` -> `SalePrice` or `Fireplaces` -> `SalePrice` shows a particular pattern with verticals lines roughly meaning that they are discrete variables with a short range but I don't know if they need some sort of "special treatment".
