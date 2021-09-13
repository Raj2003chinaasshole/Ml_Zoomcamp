import numpy as np
import pandas as pd

# Q. 1
print('numpy version -->', np.__version__)


# Q. 2
print('pandas version -->', pd.__version__)


# Q. 3
df = pd.read_csv('data.csv')
bmw = df[df['Make'] == 'BMW']
msrp = bmw['MSRP'].values.mean()
avg_price_of_bmw = round(msrp, 2)
print('Average Price of BMW -->', avg_price_of_bmw)


# Q. 4
year = df[df['Year'] >= 2015]
missing_hp = year['Engine HP'].isnull().sum()
print('Missing HP -->', missing_hp)


# Q. 5
mean_hp_before = df['Engine HP'].mean()
data_replace = df['Engine HP'].fillna(value=mean_hp_before)
mean_hp_after = df['Engine HP'].mean()
print('HP Before -->', round(mean_hp_before))
print('HP After -->', round(mean_hp_after))


# Q. 6
rolls_royce = df[df['Make'] == 'Rolls-Royce']
select_cloumns = rolls_royce[['Engine HP', 'Engine Cylinders', "highway MPG"]]
X = np.array(select_cloumns.drop_duplicates())
X_T = X.transpose()
XTX = X_T.dot(X)
inverese = np.linalg.inv(XTX)
print('Sum of Inverse -->', inverese.sum())


# Q. 7
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
multi = inverese.dot(X_T)
w = multi.dot(y)
print('First Element -->', w[0])
