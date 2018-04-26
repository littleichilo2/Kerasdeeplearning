import numpy
import pandas
import keras
import seaborn
from numpy import concatenate
from scipy import stats
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense,Activation,BatchNormalization
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def baseline_model(neurons=500,init_mode='random_uniform',binit_mode='zeros',learn_rate=0.01,momentum=0.9,decay=1e-6):#learn_rate=0.0001,momentum=0.8
	model=Sequential()
	#model.add(Dropout(0.0),input_shape=(11,))
	model.add(Dense(neurons,kernel_initializer=init_mode,bias_initializer=binit_mode,input_shape=(9,),activation='relu'))#,,input_shape=(11,)bias_constraint=maxnorm(weight_constraint),kernel_regularizer=regularizers.L1L2(0.0,0.0),bias_regularizer=regularizers.L1L2(0.0,0.0)

	#model.add(BatchNormalization())
	model.add(Dense(neurons,kernel_initializer=init_mode,bias_initializer=binit_mode,activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	#model.add(BatchNormalization())
	model.add(Dense(neurons,kernel_initializer=init_mode,bias_initializer=binit_mode,activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	#model.add(BatchNormalization())
	model.add(Dense(neurons,kernel_initializer=init_mode,bias_initializer=binit_mode,activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	model.add(Dense(neurons,kernel_initializer=init_mode,bias_initializer=binit_mode,activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	model.add(Dense(neurons,kernel_initializer=init_mode,bias_initializer=binit_mode,activation='relu'))#,kernel_constraint=maxnorm(weight_constraint)
	#model.add(Dense(neurons,kernel_initializer='uniform',activation='relu'))
	#model.add(Dense(neurons,kernel_initializer='uniform',activation='relu'))
	#model.add(Dense(neurons,kernel_initializer='uniform',activation='relu'))
	#model.add(Dense(100,kernel_initializer='uniform',activation='relu'))

	#model.add(BatchNormalization())
	model.add(Dense(1,kernel_initializer=init_mode,bias_initializer=binit_mode,activation='linear'))
	optimizer = SGD(lr=learn_rate, momentum=momentum,decay=decay)
	model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['hinge'])#'Adagrad'
	return model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#config.gpu_options.allow_growth=True
#set_session(tf.Session(config=config))
dataframe = pandas.read_csv("medium_ALL_no_address_no_period.csv", header=None,encoding="utf-8")
dataset = dataframe.values
dataset=dataset[:34840,:10]
lbencoder=LabelEncoder()
for k in range(0,10):
	dataset[:,k]=lbencoder.fit_transform(dataset[:,k])


X = dataset[:,:9]
sX=StandardScaler().fit_transform(X)
Y=dataset[:,9]
scalerY=StandardScaler().fit(Y.reshape(-1,1))
sY=scalerY.transform(Y.reshape(-1,1))
print(sY[17789:17800])

TESTX=dataset[:1000,:9]
sTESTX=sX[:10000,:]

TESTY=dataset[:10000,9]
sTESTY=sY[:10000]
#TESTY=Y[:1000]

seed = 7
numpy.random.seed(seed)
model = KerasRegressor(build_fn=baseline_model, verbose=1)


# define the checkpoint
"""filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]"""

"""
history=model.fit(sX,sY,batch_size=12000,epochs=2000,shuffle=True,validation_data=(sTESTX,sTESTY),callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')])#,callbacks=callbacks_list
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
r=pandas.DataFrame(scalerY.inverse_transform(predictions.reshape(-1,1)),TESTY)
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
batch_size = [12000,16000,10000,8000,5000,2000]#,,9985100,1000,16000,,,50016000,,20000,
epochs = [100,200,300,400,500,600,700,800,900,1000,1500]#1000,1500,2000,
#activation = ['softplus']#ELU,ThresholdedReLU,PReLU,LeakyReLU
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#reg1=[regularizers.L1L2(0.0,0.0),regularizers.L1L2(0.0,0.001),regularizers.L1L2(0.0,0.01),regularizers.L1L2(0.0,0.1),regularizers.L1L2(0.001,0.0),regularizers.L1L2(0.001,0.001),regularizers.L1L2(0.001,0.01),regularizers.L1L2(0.001,0.1),regularizers.L1L2(0.01,0.0),regularizers.L1L2(0.01,0.001),regularizers.L1L2(0.01,0.01),regularizers.L1L2(0.01,0.1),regularizers.L1L2(0.1,0.0),regularizers.L1L2(0.1,0.001),regularizers.L1L2(0.1,0.01),regularizers.L1L2(0.1,0.1)]
#reg2=[regularizers.L1L2(0.0,0.0),regularizers.L1L2(0.0,0.001),regularizers.L1L2(0.0,0.01),regularizers.L1L2(0.0,0.1),regularizers.L1L2(0.001,0.0),regularizers.L1L2(0.001,0.001),regularizers.L1L2(0.001,0.01),regularizers.L1L2(0.001,0.1),regularizers.L1L2(0.01,0.0),regularizers.L1L2(0.01,0.001),regularizers.L1L2(0.01,0.01),regularizers.L1L2(0.01,0.1),regularizers.L1L2(0.1,0.0),regularizers.L1L2(0.1,0.001),regularizers.L1L2(0.1,0.01),regularizers.L1L2(0.1,0.1)]

#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#optimizer = [ 'Adagrad']
#neurons=[1200,1100,1000,900,800,700,600,500,400,300]#,30000,12500,100003875,,,90,80,70,60,,40,30,20,10,51937,1000,500,
#neurons2=[100]#
#weight_constraint = [1, 2, 3, 4, 5]
#init_mode = ['random_uniform','random_normal', 'lecun_uniform','lecun_normal', 'normal', 'zeros', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#binit_mode=	['random_uniform','random_normal', 'lecun_uniform','lecun_normal', 'normal', 'zeros', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#dropout_rate = [0.0,0.2,0.4,0.6]#
#learn_rate = [0.001, 0.01,0.0001,0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]#
#decay=[0.0,0.1,0.01,0.001,0.0001,0.00001]

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=-1,cv=5)
grid_result = grid.fit(sX, sY)
print("Best: %f using %s" % (-grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
ranks=grid_result.cv_results_['rank_test_score']
params = grid_result.cv_results_['params']
for mean, stdev,rank, param in zip(means, stds, ranks,params):
    print("%f (%f) rank:%d with: %r" % (-mean, stdev, rank,param))



