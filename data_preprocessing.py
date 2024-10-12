import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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
df.isnull().sum() / df.shape[0] * 100

# drop useless columns
df.drop_duplicates(inplace=True)
df.drop(['county', 'url', 'region_url', 'VIN', 'size', 'image_url'], axis=1, inplace=True)
df.shape

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

# clean 'year' column and plot
# fn = lambda y: re.split(r'[.-]', y)[0]
# df.year = df.year.astype('str').apply(fn)
# df.year = pd.to_numeric(df.year, errors='coerce')

df.year.hist(bins = 50)

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
df['quantile'] = pd.qcut(df.price, 5)
df['quantile'].value_counts()

pd.pivot_table(df, index='year', columns='type', values='price', aggfunc='mean').sort_index(ascending=False)

pd.pivot_table(df, index='year', columns='type', values='price', aggfunc='count').sort_index(ascending=False).plot()

pd.pivot_table(df, index='type', columns='fuel', values='price', aggfunc='mean').plot(rot=45)

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

# check pivot table again to ensure changes
pd.pivot_table(df, index='type', columns='fuel', values='price', aggfunc='mean').plot()

# check distribution of price
df.hist('price')

# check if we remove rows that have non-null values at least for 21 columns out of 23 columns, how many rows will remain
df_temp = df.dropna(thresh=21, axis=0)
df_temp.shape

# check the outliers in odometer column
df.boxplot('odometer')
# remove outliers from odometer column using quantiles
df = df[(df.odometer < df.odometer.quantile(.995)) & (df.odometer > df.odometer.quantile(.005))]
print(df.shape)
df.boxplot('odometer')

df.hist('odometer')

# change cylinders dtype from object to numeric
df.cylinders.head()
df.cylinders.value_counts()
df.cylinders = df.cylinders.apply(lambda x: str(x).lower().replace('cylinders', '').strip())
df.cylinders = pd.to_numeric(df.cylinders, errors='coerce')
df.cylinders.dtype
# fill null values with mean value for cylinders
df.cylinders.isnull().sum()
df.cylinders = df.cylinders.fillna(df.cylinders.mean())

# fill null values with median value for years
df.hist('year')
df.year = df.year.fillna(df.year.median())

# make text in description into lower case
df.description = df.description.apply(lambda x: str(x).lower())
df.description.sample(10)

df.paint_color.mode()

df.paint_color.value_counts().plot(kind='bar')

# fill null values with mode for categorical data 
def fill_nan_with_mode(dataframe, column_name):
    return dataframe[column_name].fillna(dataframe[column_name].value_counts().index[0])
cat_variables = ['fuel', 'manufacturer', 'title_status', 'transmission', 'model']
for var in cat_variables:
    df[var] = fill_nan_with_mode(df, var)

# fill nan in lat and long variables
df.long.hist(bins=50)
df.lat.hist(bins=50)
df.long = df.long.fillna(df.long.median())
df.lat = df.lat.fillna(df.lat.mean())
# fill na in age column with median
df.age.hist()
df.age = df.age.fillna(df.age.median())

