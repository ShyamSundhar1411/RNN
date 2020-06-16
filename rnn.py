#Importing Libraries
import numpy as np
import matplotlib.pyplot as p
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
#Importing training test
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
#Feature Scaling
o = MinMaxScaler(feature_range = (0,1))
tts = o.fit_transform(training_set)
#Specific Data Structure
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(tts[i-60:i,0])
    y_train.append(tts[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#Building RNN
regression = Sequential()
#LSTM Layers
regression.add(LSTM(units = 80,return_sequences = True, input_shape = (x_train.shape[1],1)))
regression.add(Dropout(0.2))
regression.add(LSTM(units = 80 ,return_sequences = True))
regression.add(Dropout(0.2))
regression.add(LSTM(units = 80 , return_sequences = True))
regression.add(Dropout(0.2))
regression.add(LSTM(units = 80))
regression.add(Dropout(0.2))
#Output Layer
regression.add(Dense(units = 1))
#Training RNN
#regression.compile(optimizer = 'adam', loss = 'mean_squared_error')
#regression.fit(x_train,y_train,epochs = 100,batch_size = 32)
#Predicting Output
dataset = pd.read_csv('Google_Stock_Price_Test.csv')
stp = dataset.iloc[:, 1:2].values
#Equating results
conc = pd.concat((dataset_train['Open'], dataset['Open']), axis = 0)
u = conc[len(conc) - len(dataset) - 60:].values
u = u.reshape(-1,1)
u = o.transform(u)
x_test = []
for i in range(60,80):
    x_test.append(u[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
pr = regression.predict(x_test)
pr = o.inverse_transform(pr)
#Visualization
p.plot(stp, color = 'red', label = 'Real Stock Price')
p.plot(pr, color = 'blue' , label = 'Predicted Stock Price')
p.title('Google Stock Price')
p.xlabel('Time')
p.ylabel('Google Stock Price')
p.legend(loc = 'upper right')
p.show()


 




    
    
