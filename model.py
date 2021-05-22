import pandas as pd
import numpy as np
import joblib

data=pd.read_csv('dataset.csv')

x_col = [ 'followers', 'article', 'document', 'image', 'poll', 'text', 'video', 'achievement', 'call to action', 
         'insights', 'job opening', 'other', 'num_hashtags', 'num_links', 'contlen', 'conf2', 'relevance_score']
y_col = ['Reach']

X = data[x_col]
y = data.Reach
X.columns=[ 'followers', 'article', 'document', 'image', 'poll', 'text', 'video', 'achievement', 'call to action', 'insights', 'job opening', 'other', 'num_hashtags', 'num_links', 'contlen', 'conf', 'relevance_score']

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators= 2000,
 min_samples_split=10,
 min_samples_leaf= 2,
 max_features='auto',
 max_depth= 40,
 bootstrap= True,random_state=1)#final_model
 
reg.fit(X,y.values.ravel())

filename = 'model.pkl'
joblib.dump(reg, filename)

 
 


