import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import math

# read the data
path = 'vehicles.csv'
df = pd.read_csv(path)
# display the structure of the dataframe
print('Size of Dataframe: ', df.shape)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.sample(10))

print('Columns of Dataframe: \n', df.columns)

df.info()

df.describe()

# null values in dataframe
df.isnull().sum() / df.shape[0]

# display distinct values for categorical features
object_columns = ['manufacturer', 'condition', 'fuel', 'title_status',
                  'transmission', 'type', 'paint_color']
for col in object_columns:
    print('Values of Column: ', col)
    print(df[col].unique(), '\n')

# plot distribution of cars for different companies and types separately
fig, ax = plt.subplots(2, 1)
fig.set_figwidth(10)
fig.set_figheight(8)
colors = ['tab:green', 'tab:green', 'tab:green', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']
for i in range(len(df['manufacturer'].value_counts().index) - 11):
    colors.append('tab:red')
ax[0].bar(df['manufacturer'].value_counts().index, df['manufacturer'].value_counts().to_numpy(), color=colors)
ax[0].tick_params(axis='x', labelrotation=45, labelsize=9, pad=0.5)
ax[0].set_xticks(ticks=df['manufacturer'].value_counts().index, labels=df['manufacturer'].value_counts().index, horizontalalignment='right')
colors = ['tab:green', 'tab:green', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:red', 'tab:red', 'tab:red', 'tab:red', 'tab:red', 'tab:red']
ax[1].bar(df['type'].value_counts().index, df['type'].value_counts().to_numpy(), color=colors)
ax[1].tick_params(axis='x', labelrotation=45, labelsize=9, pad=0.5)
fig.tight_layout()
plt.show()

# plot a pie chart to show the proportion of cars' conditions
condition_distribution = df['condition'].value_counts().to_numpy() / df['condition'].notnull().sum()
fig, ax = plt.subplots()
explode = (0.1, 0.05, 0.05, 0.05, 0.05, 0.05)
colors = ('tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'grey', 'grey')
wedges, texts, autotexts = ax.pie(
        condition_distribution,
        explode = explode,
        colors = colors,
        labels = df['condition'].value_counts().index,
        autopct = '%1.1f%%',
        startangle = 90,
        wedgeprops = {'edgecolor': 'k', 'linewidth': 1, 'antialiased': True},
        shadow = True)
threshold = 5
for label, pct_label in zip(texts, autotexts):
    pct_value = pct_label.get_text().rstrip('%')
    if float(pct_value) < threshold:
        label.set_text('')
        pct_label.set_text('')
ax.legend(bbox_to_anchor = (1.2, 1))
ax.set_title('condition', {'fontweight': 'bold',
                           'fontsize': 14}, pad = 0)
plt.show()

# clean 'year' column and plot
# fn = lambda y: re.split(r'[.-]', y)[0]
# df.year = df.year.astype('str').apply(fn)
# df.year = pd.to_numeric(df.year, errors='coerce')

df.year.hist(bins = 50)
plt.show()

# add column 'age' - current year is 2024
df['age'] = 2024 - df['year']
print(df.loc[50:70, 'age'])
# add column 'price per mile' - it gives us an estimate of probable better sale offers
df['price_per_mile'] = df.price / df.odometer
print(df.loc[50:70, 'price_per_mile'])

# new cars i.e. age <= 10 (filtering)
new_cars = df[df['age'] <= 10]
# new cars with prices over 50000
new_cars_high_price = df[(df.age <= 10) & (df.price > 50000)]
print(new_cars.shape)
print(new_cars_high_price.shape)

# check distribution of data in intervals with same lengths (quartile)
pd.cut(df.price, 5).value_counts()

# create qunatiles for price values
df['q_intervals'] = pd.qcut(df.price, 5)
df['q_intervals'].value_counts()

pd.pivot_table(df, index='year', columns='type', values='price', aggfunc='mean').sort_index(ascending=False)

pd.pivot_table(df, index='year', columns='type', values='price', aggfunc='count').sort_index(ascending=False).plot()
plt.show()

pd.pivot_table(df, index='type', columns='fuel', values='price', aggfunc='mean').plot(rot=45)
plt.show()
# check some outlier data points
print(df[df['price'] > 1000000].price)
idx_rm = df[df['price'] > 1000000].index.tolist()
sample_idx = idx_rm.index(df[df['price'] == 5000000].index)
print(df.loc[idx_rm[sample_idx], :].description)

""" As we can see in the outliers, many of them are just successive digits or all 1s.
Also, the description for a car with 5million price indicates that it is just an ad.
We remove outliers from the dataset using quantile.
"""
df = df[(df.price < df.price.quantile(.995)) & (df.price > df.price.quantile(.005))]
print(df.shape)
df.boxplot('price')
plt.show()

# check pivot table again to ensure changes
pd.pivot_table(df, index='type', columns='fuel', values='price', aggfunc='mean').plot()

# check distribution of price
df.hist('price')
plt.show()

# check the outliers in odometer column
df.boxplot('odometer')
plt.show()
# remove outliers from odometer column using quantiles
df = df[(df.odometer < df.odometer.quantile(.995)) & (df.odometer > df.odometer.quantile(.005))]
print(df.shape)
df.boxplot('odometer')
plt.show()
df.hist('odometer')
plt.show()

# remove rows that have non-null values at least for 20 columns out of 23 columns
df = df.dropna(thresh=20, axis=0)
df.shape

# change cylinders dtype from object to numeric
df.cylinders.head()
df.cylinders.value_counts()
df.cylinders = df.cylinders.apply(lambda x: str(x).lower().replace('cylinders', '').strip())
df.cylinders = pd.to_numeric(df.cylinders, errors='coerce')
df.cylinders.dtype
# fill null values with median value for cylinders
df.cylinders.isnull().sum()
df.cylinders = df.cylinders.fillna(df.cylinders.median())

# drop rows for columns with low percentage of null values
df.isnull().sum() / df.shape[0]
df = df.dropna(subset = ['year', 'manufacturer', 'model', 'fuel', 'title_status', 'transmission'])

# check distribution of prices over states to examine importance of 'state' column
state_price = df.groupby('state')[['price']].mean()
plt.bar(state_price.index, state_price.values.reshape(len(state_price)))

# drop columns that we won't need for training
df.drop_duplicates(inplace=True)
df.drop(['id', 'county', 'url', 'region_url', 'region', 'state', 'size', 'image_url', 'lat', 'long', 'description'], axis=1, inplace=True)
df.shape


# impute missing data in categorical variables with NOT AVAILABLE
# we want our model to know the absence of data in these rows
df[['condition', 'VIN', 'drive', 'type', 'paint_color']] = \
    df[['condition', 'VIN', 'drive', 'type', 'paint_color']].fillna('n\a')
df.isnull().sum() / df.shape[0]
df.head(10)

# change VIN number to has_vin or no_vin 
df['VIN'] = df['VIN'].apply(lambda x: 'has_vin' if x != 'n\a' else 'no_vin')

# correct 'posting_date' datatype
df.posting_date.head(10)
df.posting_date = pd.to_datetime(df.posting_date, utc=True)
df.posting_date = df.posting_date.dt.date
df.posting_date.head(10)

# make all categorical variables lower case
for col in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', \
            'transmission', 'drive', 'type', 'paint_color']:
    df[col] = df[col].apply(lambda x: str(x).lower())


# check final dataset to ensure it is prepared for training model
df.isnull().any()
numeric = df._get_numeric_data()
corrdata = numeric.corr()
ax = sns.heatmap(
    corrdata,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200)
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()


"""
    TRAINING TIME!
"""

# we use OLS to train a linear regression model
# add a constant column to calculate model's intercept
df['constant'] = 1
X1 = df[['constant', 'odometer', 'age', 'cylinders']]
y1 = df['price']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=3)
reg1 = sm.OLS(y1_train, X1_train).fit()
reg1.summary()

y1_pred = reg1.predict(X1_test)
rmse1 = math.sqrt(mean_squared_error(y1_pred, y1_test))
print('Root Mean Squared Error: ', rmse1)

# VIF statistics should be under 4 to ensure the variables are not highly correlated 
pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)


# now include categorical variables in the training data
X2 = pd.get_dummies(df[['constant', 'odometer', 'age', 'cylinders',\
                       'type', 'VIN', 'condition', 'fuel']])
y2 = df['price'].values
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2.astype(float), y2, test_size=0.3, random_state=3)
reg2 = sm.OLS(y2_train, X2_train).fit()
reg2.summary()

y2_pred = reg2.predict(X2_test)
rmse2 = math.sqrt(mean_squared_error(y2_pred, y2_test))
print('Root Mean Squared Error: ', rmse2)

plt.scatter(y2_pred, y2_test)
plt.show()

# check distribution of residuals
residuals = y2_pred.to_numpy() - y2_test
plt.hist(residuals, bins=50)
plt.show()


# cross validation
X3 = pd.get_dummies(df[['odometer', 'age', 'cylinders',\
                       'type', 'VIN', 'condition', 'fuel']])
y3 = df.price.values

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3.astype(float), y3, test_size=0.3, random_state=3)

reg3 = LinearRegression().fit(X3_train, y3_train)
reg3.score(X3_train, y3_train)

reg3 = LinearRegression()
scores = cross_val_score(reg3, X3, y3, cv=5, scoring='neg_mean_squared_error')
np.sqrt(np.abs(scores))

# non-linear regression 
X4 = df[['odometer', 'age', 'cylinders']]
y4 = df.price.values

for i in range(X4.shape[1]):
    fig, ax = plt.subplots()
    ax.scatter(X4.iloc[:, i].values, y4)
    plt.show()

# feature engineering - create new features by multiplying pairs of two variables
degree = 2
p = preprocessing.PolynomialFeatures(degree).fit(X4)
print(p.get_feature_names_out(X4.columns))
X4 = preprocessing.PolynomialFeatures(degree).fit_transform(X4)

X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.3, random_state=3)

reg4 = LinearRegression().fit(X4_train, y4_train)
y4_test_pred = reg4.predict(X4_test)
rmse4 = math.sqrt(mean_squared_error(y4_test_pred, y4_test))

print('Root Mean Squared Error: ', rmse4)


# Train a random forest regressor to 
X5 = pd.get_dummies(df[['odometer', 'age', 'cylinders', 'type', 
                        'fuel', 'condition', 'VIN']])
y5 = df.price.values
# split data into training and test dataset
X5_train, X5_test, y5_train, y5_test = train_test_split(
    X5, y5, test_size=0.3, random_state=3)
# train RandomForestRegressor 
rfr = RandomForestRegressor(oob_score=True, random_state=3)
rfr.fit(X5_train, y5_train)
# get the R^2 score and OOB score for our model
print("R^2 score: ", rfr.score(X5_test, y5_test))
print("OOB Score: ", rfr.oob_score_)

# perform 5fold corss validation to evaluate the performance of our model
rfr_cv = RandomForestRegressor(oob_score=True, random_state=3)
scores = cross_val_score(rfr_cv, X5, y5, cv=5, scoring='neg_mean_squared_error')
scores = np.sqrt(np.abs(scores))
print(f"5-fold cross validation results:\nmean value: {scores.mean()}\nstandard deviation: {scores.std()}")

# paramter optimization using GridSearchCV
params = {'max_features': [0.3, 0.5, 1.0], 'min_samples_leaf': [1, 2, 3],
              'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 3, 4]}
rfr_gsearch = RandomForestRegressor(oob_score=True, random_state=3)
rfr_gs = GridSearchCV(estimator=rfr_gsearch, param_grid=params, 
                             scoring='neg_mean_squared_error', cv=2, n_jobs=-1)
rfr_gs.fit(X5_train, y5_train)
print("best parameters for our randomforest model: ", rfr_gs.best_params_)
print("best rmse after grid search: ", np.sqrt(np.abs(rfr_gs.best_score_)))
rfr_tuned = rfr_gs.best_estimator_
print("best R^2 score: ", rfr_tuned.score(X5_test, y5_test))
print("best oob score: ", rfr_tuned.oob_score_)
y5_pred = rfr_tuned.predict(X5_test)
print("RMSE for test data:, ", math.sqrt(mean_squared_error(y5_pred, y5_test)))

plt.scatter(y5_pred, y5_test)
plt.show()







