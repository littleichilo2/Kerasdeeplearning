import pandas
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.externals import joblib
import tensorflow as tf
dataframe = pandas.read_csv("medium_ALL_no_address.csv", header=None,encoding="utf-8")
dataset = dataframe.values
#dataset=dataset[:34830,:11]
lbencoder=LabelEncoder()
for k in range(0,10):
	dataset[:,k]=lbencoder.fit_transform(dataset[:,k])

X = dataset[:34830,:10]
sX=StandardScaler().fit_transform(X)
Y=dataset[:34830,10]
scalerY=StandardScaler().fit(Y.reshape(-1,1))
sY=scalerY.transform(Y.reshape(-1,1)).reshape(34830,)

"""
sTESTX=sX[33830:,:]
TESTY=dataset[33830:,10]
sTESTY=sY[33830:]
seed = 7

#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#model=GradientBoostingRegressor(n_estimators=15,learning_rate=0.1,max_features=3,max_depth=3,random_state=seed,loss='ls',min_samples_split=30)
#model = DecisionTreeRegressor()
model=KNeighborsRegressor(n_neighbors=10,weights='distance')

#model=SGDRegressor(alpha=0.1, average=False, epsilon=0.1, eta0=0.01,fit_intercept=True, l1_ratio=0.5, learning_rate='optimal',loss='squared_loss', max_iter=None, n_iter=None, penalty='l2',power_t=0.25, random_state=None, shuffle=True, tol=None, verbose=1,warm_start=False)
#scoring = 'neg_mean_squared_error'
#results = model_selection.cross_val_score(model, sX, sY, cv=kfold,scoring=scoring)
#print(results.mean())
model.fit(sX,sY)

print(mean_squared_error(sTESTY,model.predict(sTESTX)))

predictions=model.predict(sTESTX)

r=pandas.DataFrame(scalerY.inverse_transform(predictions.reshape(-1,1)),TESTY)
r.to_csv('pp.csv')
joblib.dump(model,'filename.pkl')
"""

parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'max_features'      : [3, 5, 10],
        'random_state'      : [7],
        'learning_rate'		: [0.1,0.01,0.001],
        #'n_jobs'            : [-1],
        #'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [1,3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
}
parameters2 = {
        'loss'      : ['squared_loss'],#, 'huber','epsilon_insensitive'
        'penalty'      : ['l2'],#,'l1','elasticnet'
        'alpha'      : [0.0001,0.001,0.01,0.1],
        'learning_rate':['optimal'],
        'l1_ratio'		: [0.15,0.1,0.3,0.4,0.5],
        #'n_jobs'            : [-1],
        'fit_intercept' : [True],
        'verbose':[1],
}
clf = model_selection.GridSearchCV(GradientBoostingRegressor(), parameters)
clf.fit(sX,sY)
 
print(clf.best_estimator_)

#print(model.feature_importances_)
#print(model.mean())