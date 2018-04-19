import pandas
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.linear_model import SGDRegressor
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth=True
dataframe = pandas.read_csv("medium_ALL_no_address.csv", header=None,encoding="shift-jis")
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

"""
sTESTX=sX[20000:30000,:]
sTESTY=sY[20000:30000]
seed = 7
num_trees = 300
max_features=3
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
       fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
       loss='squared_loss', max_iter=None, n_iter=None, penalty='l2',
       power_t=0.25, random_state=None, shuffle=True, tol=None,
       verbose=0, warm_start=False)
#results = model_selection.cross_val_score(model, sX, sY, cv=kfold)
model.fit(sX,sY)
"""

parameters = {
        'learning_rate':['invscaling','constant','optimal'],
        'panalty':['none', 'l2', 'l1','elasticnet'],
}
clf = model_selection.GridSearchCV(SGDRegressor(), parameters)
clf.fit(sX,sY)
 
print(clf.best_estimator_)

#print(model.feature_importances_)
#print(model.mean())