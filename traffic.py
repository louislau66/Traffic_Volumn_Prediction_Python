import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import interactive
interactive(False)

"""
Data Set Information:

Hourly Interstate 94 Westbound traffic volume for MN DoT ATR station 301, roughly midway between Minneapolis and St Paul, MN. Hourly weather features and holidays included for impacts on traffic volume.


Attribute Information:

holiday Categorical US National holidays plus regional holiday, Minnesota State Fair
temp Numeric Average temp in kelvin
rain_1h Numeric Amount in mm of rain that occurred in the hour
snow_1h Numeric Amount in mm of snow that occurred in the hour
clouds_all Numeric Percentage of cloud cover
weather_main Categorical Short textual description of the current weather
weather_description Categorical Longer textual description of the current weather
date_time DateTime Hour of the data collected in local CST time
traffic_volume Numeric Hourly I-94 ATR 301 reported westbound traffic volume
"""

df=pd.read_csv(r"...Metro_Interstate_Traffic_Volume.csv")
pd.set_option('display.max_columns',50)

#heatmap of missing values
sns.set_palette('GnBu_d')
sns.set_style('whitegrid')
sns.heatmap(df.isnull())
plt.show()
#data pre-processing
df.columns
df.info()
df.head()
#unique values for object columns
df['holiday'].unique()
df['weather_main'].unique()
df['weather_description'].unique()
#data distribution for data columns
df[['temp','rain_1h','snow_1h','clouds_all','traffic_volume']].describe()
#fill temp=0 with mean value
pd.value_counts(df['temp']==0)
tempmean=df.temp.mean()
for i in range(0,len(df)):
    if df.temp[i]==0:
        df.temp[i]=tempmean

sns.distplot(df['traffic_volume'])
plt.show()
print(df[df['traffic_volume']==max(df['traffic_volume'])])

sns.pairplot(df)
plt.show()

#get day_of_week and time from date_time
df['date_time'] = pd.to_datetime(df['date_time'])
df['day_of_week'] = df['date_time'].dt.day_name()
df['time'] = df['date_time'].dt.time
df['month']=df['date_time'].dt.month
df.drop('date_time',axis=1,inplace=True)
df.info()

#df.drop(['day_of_week','time'],axis=1,inplace=True)
df1=pd.get_dummies(df,columns=['holiday',  'weather_main','weather_description','day_of_week', 'time','month'],drop_first=True)
df1.info()
X=df1.copy()
X.drop('traffic_volume',axis=1,inplace=True)
y=df1['traffic_volume']


sns.heatmap(df.corr(),annot=True)
plt.xticks(rotation=20)
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=101)
#
X_trainnew = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)

import statsmodels.formula.api as sm
#lst = list(range(0,len(X_trainnew)))
leng=X_trainnew.shape[1]
X_opt=list(range(0,leng))
regressor = sm.OLS(y_train, X_trainnew[:,X_opt]).fit()
print(regressor.summary())

#for attr in dir(regressor):
 #   if not attr.startswith('_'):
  #      print(attr)

#Backward Feature Elimination 
pvalues=regressor.pvalues
#create dictionary for reference from pvalues to column name
col=list(X_train.columns)
col.insert(0,'const')
ind=['x%d' % i for i in range(1, 103)]
ind.insert(0,'const')
zipobj=zip(ind,col)
col=dict(zipobj)
print(col)
print(X.columns)

length=len(pvalues)
pmax=1
while pmax>0.05:
    ind=pvalues.idxmax() 
    pvalues=pvalues.drop(index=ind)
    pmax=max(pvalues)
    X.drop(col[ind],axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=101)
    X_trainnew = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
    leng=X_trainnew.shape[1]
    X_opt=list(range(0,leng))
    regressor = sm.OLS(y_train, X_trainnew[:,X_opt]).fit()
    pvalues=regressor.pvalues
    col=list(X_train.columns)
    col.insert(0,'const')
    ind=['x%d' % i for i in range(1, len(col))]
    ind.insert(0,'const')
    zipobj=zip(ind,col)
    col=dict(zipobj)


#Rebuild model with selected features
final_col=list(col.values())
X_final=X[final_col[1:]]

X_train, X_test, y_train, y_test = train_test_split(X_final,y,test_size=1/3,random_state=101)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lm=LinearRegression()
lm.fit(X_train, y_train)
y_pred=lm.predict(X_test)
r2_test=r2_score(y_pred,y_test)

y_trainpred=lm.predict(X_train)
r2_train=r2_score(y_train,y_trainpred)

print(regressor.summary())
print('X columns: ', col)

#plot predicted & test values
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('Predicted y')
plt.show()

#print('Slope:', lm.coef_)
#print('Intercept:', lm.intercept_)
print('R2 test: ', r2_test)
print('R2 train: ', r2_train)
from sklearn import metrics
print('MSE: ', metrics.mean_squared_error(y_test,y_pred))



plt.boxplot(x=y_pred)
plt.show()

#K-fold cross validation
from sklearn.model_selection import cross_val_score
clf = LogisticRegression()
cross_val_score(clf,X,y,cv=4).mean()

