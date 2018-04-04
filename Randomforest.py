import pandas
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
dataframe = pandas.read_csv("BIG_ALL_period_no_address.csv", header=None,encoding="shift-jis")
dataset = dataframe.values
dataset=dataset[:117698,:11]
lbencoder=LabelEncoder()
for k in range(0,11):
	dataset[:,k]=lbencoder.fit_transform(dataset[:,k])

X = dataset[:,:10]
sX=StandardScaler().fit_transform(X)
Y=dataset[:,10]
scalerY=StandardScaler().fit(Y.reshape(-1,1))
sY=scalerY.transform(Y.reshape(-1,1)).reshape(117696,)


sTESTX=sX[50000:60000,:]
sTESTY=sY[50000:60000]
seed = 7
num_trees = 50
max_features=3
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestRegressor(n_estimators=num_trees, max_features=max_features,min_samples_split=30,max_depth=25,random_state=7)
#results = model_selection.cross_val_score(model, sX, sY, cv=kfold)
model.fit(sX,sY)
"""

parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'max_features'      : [3, 5, 10],
        'random_state'      : [7],
        'n_jobs'            : [1],
        'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
}
clf = model_selection.GridSearchCV(RandomForestRegressor(), parameters)
clf.fit(sX,sY)
 
print(clf.best_estimator_)
"""
print(model.feature_importances_)
#print(model.mean())