import numpy as np
import pandas as pd

# Q. 1
print(np.__version__)


# Q. 2
print(pd.__version__)


# Q. 3
df = pd.read_csv('data.csv')
bmw = df[df['Make'] == 'BMW']
msrp = bmw['MSRP'].values.mean()
avg_price_of_bmw = round(msrp, 2)
print(avg_price_of_bmw)


# Q. 4
year = df[df['Year'] >= 2015]
missing_hp = year['Engine HP'].isnull().sum()
print(missing_hp)


# Q. 5
mean_hp_before = df['Engine HP'].mean()
data_replace = df['Engine HP'].fillna(value=mean_hp_before)
mean_hp_after = df['Engine HP'].mean()
print(round(mean_hp_before))
print(round(mean_hp_after))


# Q. 6
rolls_royce = df[df['Make'] == 'Rolls-Royce']
select_cloumns = rolls_royce[['Engine HP', 'Engine Cylinders', "highway MPG"]]
X = np.array(select_cloumns.drop_duplicates())
X_T = X.transpose()
XTX = X_T.dot(X)
inverese = np.linalg.inv(XTX)
print(inverese.sum())


# Q. 7
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
multi = inverese.dot(X_T)
w = multi.dot(y)
print(w[0])
