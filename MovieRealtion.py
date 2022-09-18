import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None

# read data
df = pd.read_csv('movies.csv')
df.head()

# data cleaning
df.budget = df['budget'].fillna(0)
df.gross = df['gross'].fillna(0)
df.budget = df['budget'].astype('int32')
df.gross = df['gross'].astype('int32')
df = df.drop_duplicates()
df = df.sort_values(by='gross', inplace=False, ascending=False)
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
    
# plt the data
sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
for col_name in df.columns:
    if df[col_name].dtype == 'object':
        df[col_name] = df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
correlation_mat = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

plt.title("Correlation matrix for Numeric Features")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()

corr_pairs = correlation_mat.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]
print(strong_pairs)
