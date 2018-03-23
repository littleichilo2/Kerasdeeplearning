import numpy
import pandas
import keras
from numpy import concatenate
from scipy import stats
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense,Activation,BatchNormalization
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def baseline_model(init_mode='uniform'):#
	model=Sequential()
	#model.add(Dropout(dropout_rate))
	model.add(Dense(900,kernel_initializer='uniform',input_shape=(11,),kernel_constraint=maxnorm(1),activation='relu'))#,,input_shape=(11,)bias_regularizer=regularizers.L1L2(0.0,0.001),kernel_regularizer=regularizers.L1L2(0.0,0.0),bias_regularizer=regularizers.L1L2(0.0,0.0)

	#model.add(BatchNormalization())
	model.add(Dense(900,kernel_initializer='uniform',kernel_constraint=maxnorm(1),activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	#model.add(BatchNormalization())
	model.add(Dense(900,kernel_initializer='uniform',kernel_constraint=maxnorm(1),activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	#model.add(BatchNormalization())
	model.add(Dense(900,kernel_initializer='uniform',kernel_constraint=maxnorm(1),activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	model.add(Dense(900,kernel_initializer='uniform',kernel_constraint=maxnorm(1),activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	model.add(Dense(900,kernel_initializer='uniform',kernel_constraint=maxnorm(1),activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	#model.add(Dense(neurons,kernel_initializer='uniform',activation='relu'))
	#model.add(Dense(neurons,kernel_initializer='uniform',activation='relu'))
	#model.add(Dense(100,kernel_initializer='uniform',activation='relu'))
	#model.add(Dense(100,kernel_initializer='uniform',activation='relu'))

	#model.add(BatchNormalization())
	model.add(Dense(1,kernel_initializer='uniform',activation='linear'))
	#optimizer = SGD(lr=0.0001, momentum=0.8)
	model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mse'])#'Adagrad'
	return model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
dataframe = pandas.read_csv("BIG_ALL_period.csv", header=None,encoding="shift-jis")
dataset = dataframe.values
dataset=dataset[:117698,:12]
#dataset=stats.boxcox(dataset)
lbencoder=LabelEncoder()
for k in range(0,12):
	dataset[:,k]=lbencoder.fit_transform(dataset[:,k])



scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(dataset)
"""
scaler=StandardScaler()
scaler=scaler.fit(dataset)
scaled=scaler.transform(dataset)
"""
# split into input (X) and output (Y) variables
X = scaled[:,:11]
sX=StandardScaler().fit_transform(X)
ssX=sX[:19000,:]
Y=scaled[:,11]
scalerY=StandardScaler().fit(Y.reshape(-1,1))
sY=scalerY.transform(Y.reshape(-1,1))
ssY=sY[:19000]
#Y =StandardScaler().fit_transform(dataset[:,6].reshape(-1,1))

TESTX=scaled[:1000,:11]
sTESTX=sX[:1000,:]

TESTY=scaled[:1000,11]
sTESTY=sY[:1000]
#TESTY=Y[:1000]

seed = 7
numpy.random.seed(seed)
model = KerasRegressor(build_fn=baseline_model, verbose=1)


# define the checkpoint
"""filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]"""

"""
history=model.fit(sX,sY,batch_size=50000,epochs=1000,shuffle=True,validation_data=(sTESTX,sTESTY))#,callbacks=callbacks_list
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()

predictions=model.predict(sTESTX)
#rr=concatenate((TESTX,scalerY.inverse_transform(predictions).reshape(1000,1)), axis=1)
#r=pandas.DataFrame(scaler.inverse_transform(rr))
#r.to_csv('pp2.csv',header=None)
r=pandas.DataFrame(scalerY.inverse_transform(predictions),TESTY)
#r=pandas.DataFrame(predictions,TESTY)
r.to_csv('pp.csv')

model_json = model.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.model.save_weights("model.h5")
print("Saved model to disk")

"""

#PReLU = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
#LeakyReLU=keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
#ELU=keras.layers.advanced_activations.ELU(alpha=1.0)
#ThresholdedReLU=keras.layers.advanced_activations.ThresholdedReLU(theta=1.0)
batch_size = [50000]#,,9985,7000,6000,8000100,1000,16000,12000,,,50016000,,20000,16000
epochs = [1000]#
#activation = ['softplus']#ELU,ThresholdedReLU,PReLU,LeakyReLU
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#reg=[regularizers.L1L2(0.0,0.0),regularizers.L1L2(0.0,0.001),regularizers.L1L2(0.0,0.01),regularizers.L1L2(0.0,0.1),regularizers.L1L2(0.001,0.0),regularizers.L1L2(0.001,0.001),regularizers.L1L2(0.001,0.01),regularizers.L1L2(0.001,0.1),regularizers.L1L2(0.01,0.0),regularizers.L1L2(0.01,0.001),regularizers.L1L2(0.01,0.01),regularizers.L1L2(0.01,0.1),regularizers.L1L2(0.1,0.0),regularizers.L1L2(0.1,0.001),regularizers.L1L2(0.1,0.01),regularizers.L1L2(0.1,0.1)]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#optimizer = [ 'Adagrad']
#neurons=[1000,900,800,700,600,500,400,300,200]#,30000,12500,100003875,,,90,80,70,60,,40,30,20,10,51937,1000,500,
#neurons2=[100]#
#weight_constraint = [1, 2, 3, 4, 5]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#dropout_rate = [0.0,0.2,0.4,0.6]#
#learn_rate = [0.001, 0.01,0.0001,0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]#

param_grid = dict(batch_size=batch_size, epochs=epochs,init_mode=init_mode)
grid = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=1,cv=5,scoring='neg_mean_absolute_error')
grid_result = grid.fit(sX, sY)
print("Best: %f using %s" % (-grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
ranks=grid_result.cv_results_['rank_test_score']
params = grid_result.cv_results_['params']
for mean, stdev,rank, param in zip(means, stds, ranks,params):
    print("%f (%f) rank:%d with: %r" % (-mean, stdev, rank,param))



