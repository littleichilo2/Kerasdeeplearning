# Stochastic Gradient Boosting Classification
import pandas
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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
num_trees = 500
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingRegressor(n_estimators=num_trees, random_state=seed)
model.fit(sX,sY)

#results = model_selection.cross_val_score(model, sX, sY, cv=kfold)
"""
model = GradientBoostingRegressor()
param_grid = dict(num_trees=num_trees)
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
grid_search = model_selection.GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(sX, sY)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot
pyplot.errorbar(num_trees, means, yerr=stds)
pyplot.title("model loss")
pyplot.xlabel('trees')
pyplot.ylabel('Log Loss')
#pyplot.savefig('subsample.png')
pyplot.show()
"""
print(model.feature_importances_)
#print(results.mean())