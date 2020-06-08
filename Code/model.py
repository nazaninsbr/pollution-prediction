from keras.models import Sequential
from keras.models import Model as keras_Model
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, Input, Average
import matplotlib.pyplot as plt 
import time 
import pandas as pd
import numpy as np

class Model:
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size,
                file_save_name):

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.loss_function = loss_function 
        self.optimizer = optimizer
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.add_dropout = add_dropout

        self.original_file_save_name = file_save_name
        self.file_save_name = self.original_file_save_name 

        self.model = None 

    def plot_accuracy_and_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.file_save_name+'_loss.png')
        plt.cla()


    def actual_vs_predicted_values_visualization(self, actual, predicted):
        # scatter
        plt.scatter(actual['train'], predicted['train'])
        plt.title('train actual vs predicted')
        plt.ylabel('predicted values')
        plt.xlabel('actual values')
        plt.savefig(self.file_save_name+'_train_compare_scatter.png')
        plt.cla()
        plt.scatter(actual['test'], predicted['test'])
        plt.title('test actual vs predicted')
        plt.ylabel('predicted values')
        plt.xlabel('actual values')
        plt.savefig(self.file_save_name+'_test_compare_scatter.png')
        plt.cla()
        # time series
        data = {
            'A': [x[0] for x in np.append(actual['train'], actual['test'], axis = 0)],
            'P': [x[0] for x in np.append(predicted['train'], predicted['test'], axis = 0)]
        }

        plt.plot(data['P'], color= 'red')
        plt.plot(data['A'], color='green')
        plt.title('actual (green) vs predicted (red)')
        plt.savefig(self.file_save_name+'_compare_timeseries.png')
        plt.cla()

        # subplots 
        fig, axs = plt.subplots(2)
        fig.suptitle('actual (green) vs predicted (red)')
        axs[0].plot(data['P'], color= 'red')
        axs[1].plot(data['A'], color='green')
        plt.savefig(self.file_save_name+'_subplots_compare_timeseries.png')
        axs[0].cla()
        axs[1].cla()
        plt.close('all')

    def train_and_report_results(self):
        training_start_time = time.time()
        
        history = self.model.fit(self.train_X, self.train_y, epochs=self.number_of_epochs, batch_size=self.batch_size, validation_split=self.validation_split, verbose=2, shuffle=False)
        
        training_end_time = time.time()
        print('Training Duration:', training_end_time-training_start_time)

        self.plot_accuracy_and_loss(history)

        train_yhat = self.model.predict(self.train_X)
        test_yhat = self.model.predict(self.test_X)
        actual = {'train': self.train_y, 'test': self.test_y}
        predicted = {'train': train_yhat, 'test': test_yhat}
        self.actual_vs_predicted_values_visualization(actual, predicted)


    def test_different_loss_and_optimization_functions(self):
        for opt in ['adam', 'rmsprop', 'adagrad']:
            for l in ['mae', 'mean_squared_error']:
                self.loss_function = l
                self.optimizer = opt 
                print('Testing: Optimizer = {}, Loss = {}'.format(self.optimizer, self.loss_function))
                self.file_save_name = self.original_file_save_name+'_opt_{}_loss_{}'.format(self.optimizer, self.loss_function)
                self.train_and_report_results()

class LSTM_Model(Model):
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size, file_save_name):

        Model.__init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size, file_save_name)

        self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(30, return_sequences= True, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(LSTM(30))
        if self.add_dropout:
            model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        self.model = model

class GRU_Model(Model):
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size, file_save_name):

        Model.__init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size, file_save_name)

        self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(GRU(30, return_sequences= True, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(GRU(30))
        if self.add_dropout:
            model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        self.model = model


class RNN_Model(Model):
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size, file_save_name):

        Model.__init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size, file_save_name)

        self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(SimpleRNN(30, return_sequences= True, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(SimpleRNN(30))
        if self.add_dropout:
            model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        self.model = model


class FusionModel(Model):
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split, add_dropout,
                number_of_epochs, batch_size,
                file_save_name):

        self.train_X_1 = train_X[0]
        self.train_X_2 = train_X[1]
        self.train_X_3 = train_X[2]
        
        self.train_y = train_y

        self.test_X_1 = test_X[0]
        self.test_X_2 = test_X[1]
        self.test_X_3 = test_X[2]

        self.test_y = test_y

        self.loss_function = loss_function 
        self.optimizer = optimizer
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.add_dropout = add_dropout

        self.original_file_save_name = file_save_name
        self.file_save_name = self.original_file_save_name 

        self.model = self.create_model() 

    def create_model(self):
        print(self.train_X_1.shape, self.train_X_3.shape, self.train_X_2.shape)
        inp1 = Input(shape=(self.train_X_1.shape[1], self.train_X_1.shape[2]))
        m1 = LSTM(30, return_sequences= True)(inp1)
        m2 = LSTM(30)(m1)
        d1 = Dense(1)(m2)

        inp2 = Input(shape=(self.train_X_2.shape[1], self.train_X_2.shape[2]))
        m3 = LSTM(30, return_sequences= True)(inp2)
        m4 = LSTM(30)(m3)
        d2 = Dense(1)(m4)


        inp3 = Input(shape=(self.train_X_3.shape[1], self.train_X_3.shape[2]))
        m5 = LSTM(30, return_sequences= True)(inp3)
        m6 = LSTM(30)(m5)
        d3 = Dense(1)(m6)

        avg_l = Average()([d1, d2, d3])

        model = keras_Model(inputs=[inp1, inp2, inp3], outputs=avg_l)
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        return model 

    def train_fusion_model(self):
        training_start_time = time.time()
        
        history = self.model.fit([self.train_X_1, self.train_X_2, self.train_X_3], self.train_y, epochs=self.number_of_epochs, batch_size=self.batch_size, validation_split=self.validation_split, verbose=2, shuffle=False)
        
        training_end_time = time.time()
        print('Training Duration:', training_end_time-training_start_time)

        self.plot_accuracy_and_loss(history)

        train_yhat = self.model.predict([self.train_X_1, self.train_X_2, self.train_X_3])
        test_yhat = self.model.predict([self.test_X_1, self.test_X_2, self.test_X_3])
        actual = {'train': self.train_y, 'test': self.test_y}
        predicted = {'train': train_yhat, 'test': test_yhat}
        self.actual_vs_predicted_values_visualization(actual, predicted)