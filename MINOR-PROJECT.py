get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# READING THE DATASET
df=pd.read_csv("file2.csv")
df.head(10)

# DATATYPES IN THE DATASET
df.dtypes

df.describe()

df.info()

# DELETING THE 'HANDLE & FRIENDS_COUNT' COLUMN FROM THE DATASET
df=df.drop("handle",axis=1)
df=df.drop("friends_count",axis=1)
df.head()

# # Visualizing the data

df.plot(x='problem_count',y="max_rating",kind="scatter");

df.plot(x='friends_count',y="max_rating",kind="scatter");

df['friends_count'].plot.hist(bins = [0,50,100,200,300,400,500,600,700]);

df['contest_count'].plot.hist();

df.plot(x='problem_count',y="rating",kind="scatter");

df.plot(x='problem_count',y="friends_count",kind="scatter");

plt.figure(figsize=(10, 6))
sns.boxplot(x=df["rating"])

plt.figure(figsize=(10, 6))
sns.boxplot(x=df["max_rating"])


# Outliers in Rating
df['rating'].plot.hist(bins=50)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Graph')

# Finding the Min and Max Value
mi=df['rating'].min()
mx=df['rating'].max()
print('min:',mi)
print('max:',mx)

df['rating'].describe()

# Plotting the Bell Curve
from scipy.stats import norm
df['rating'].plot.hist(bins=50,density=True)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Graph')
rng=np.arange(df['rating'].min(),df['rating'].max())
plt.plot(rng,norm.pdf(rng,df['rating'].mean(),df['rating'].std()))

df['zscore']=((df.rating-df.rating.mean()))/df.rating.std()
df.head()

df[df['zscore']>3]


df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df

# Outliers in Friends_Count
df['friends_count'].plot.hist(bins=20)
plt.xlabel('friends_count')
plt.ylabel('Count')
plt.title('friends_count Graph')

df['friends_count'].describe()


# Plotting the Bell Curve
from scipy.stats import norm
df['friends_count'].plot.hist(bins=20,density=True)
plt.xlabel('friends_count')
plt.ylabel('Count')
plt.title('friends_count Graph')
rng=np.arange(df['friends_count'].min(),df['friends_count'].max())
plt.plot(rng,norm.pdf(rng,df['friends_count'].mean(),df['friends_count'].std()))


df['zscore']=((df.friends_count-df.friends_count.mean()))/df.friends_count.std()
df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df

# Outliers in problem_count
df['problem_count'].plot.hist(bins=20)
plt.xlabel('problem_count')
plt.ylabel('Count')
plt.title('problem_count Graph')

# Plotting the Bell Curve
from scipy.stats import norm
df['problem_count'].plot.hist(bins=20,density=True)
plt.xlabel('problem_count')
plt.ylabel('Count')
plt.title('problem_count Graph')
rng=np.arange(df['problem_count'].min(),df['problem_count'].max())
plt.plot(rng,norm.pdf(rng,df['problem_count'].mean(),df['problem_count'].std()))

df['zscore']=((df.problem_count-df.problem_count.mean()))/df.problem_count.std()
df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df


# Outliers in max_rating
df['max_rating'].plot.hist(bins=20)
plt.xlabel('max_rating')
plt.ylabel('Count')
plt.title('max_rating Graph')


# Plotting the Bell Curve
from scipy.stats import norm
df['max_rating'].plot.hist(bins=20,density=True)
plt.xlabel('max_rating')
plt.ylabel('Count')
plt.title('max_rating Graph')
rng=np.arange(df['max_rating'].min(),df['max_rating'].max())
plt.plot(rng,norm.pdf(rng,df['max_rating'].mean(),df['max_rating'].std()))


df['zscore']=((df.max_rating-df.max_rating.mean()))/df.max_rating.std()
df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df


# # MODEL

# ## Random Forest

X=df.iloc[:,:-1].values
Y=df.iloc[:,3]
Y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
 
lin.fit(x_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(x_train)

from sklearn import linear_model
regression = linear_model.LinearRegression()
model = regression.fit(x_train, y_train)
score = model.score(x_test, y_test)


pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
