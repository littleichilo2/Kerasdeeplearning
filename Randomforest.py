import pandas
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth=True
dataframe = pandas.read_csv("medium_ALL_no_address.csv", header=None,encoding="utf-8")
dataset = dataframe.values
dataset=dataset[:34830,:11]
lbencoder=LabelEncoder()
for k in range(0,11):
	dataset[:,k]=lbencoder.fit_transform(dataset[:,k])

X = dataset[:,:10]
sX=StandardScaler().fit_transform(X)
Y=dataset[:,10]
scalerY=StandardScaler().fit(Y.reshape(-1,1))
sY=scalerY.transform(Y.reshape(-1,1)).reshape(34830,)


sTESTX=sX[33830:,:]
TESTY=dataset[33830:,10]
sTESTY=sY[33830:]
seed = 7
num_trees = 200
max_features=10
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestRegressor(n_estimators=num_trees, max_features=max_features,min_samples_split=30,max_depth=30,random_state=7,oob_score=True,verbose=1,n_jobs=-1,warm_start=True)
#results = model_selection.cross_val_score(model, sX, sY, cv=kfold)
model.fit(sX,sY)
predictions=model.predict(sTESTX)

r=pandas.DataFrame(scalerY.inverse_transform(predictions.reshape(-1,1)),TESTY)
r.to_csv('pp.csv')
"""

parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 200, 300],
        'max_features'      : [10],
        'random_state'      : [7],
        'n_jobs'            : [-1],
        'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50]

}
clf = model_selection.GridSearchCV(RandomForestRegressor(oob_score=True,warm_start=True,n_jobs=-1,verbose=1), parameters)
clf.fit(sX,sY)

print(clf.best_estimator_)
"""
#print(model.feature_importances_)
#print(model.mean())