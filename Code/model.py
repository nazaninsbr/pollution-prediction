from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt 
import time 
import pandas as pd
import numpy as np

class Model:
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split,
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

        self.file_save_name = file_save_name

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
        plt.cla()

class LSTM_Model(Model):
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split,
                number_of_epochs, batch_size, file_save_name):

        Model.__init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, validation_split,
                number_of_epochs, batch_size, file_save_name)

        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(30, return_sequences= True, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(LSTM(30))
        model.add(Dense(1))
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        return model 

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

    