from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt 
import time 

class Model:
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, 
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

        self.file_save_name = file_save_name

    def plot_accuracy_and_loss(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.file_save_name+'_accuracy.png')
        plt.cla()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.file_save_name+'_loss.png')
        plt.cla()

class LSTM_Model(Model):
    def __init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, 
                number_of_epochs, batch_size, file_save_name):

        Model.__init__(self, train_X, train_y, test_X, test_y, 
                loss_function, optimizer, 
                number_of_epochs, batch_size, file_save_name)

        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics = ['accuracy'])
        return model 

    def train_and_report_results(self):
        training_start_time = time.time()
        history = self.model.fit(self.train_X, self.train_y, epochs=self.number_of_epochs, batch_size=self.batch_size, validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
        training_end_time = time.time()
        print('Training Duration:', training_end_time-training_start_time)
        self.plot_accuracy_and_loss(history)

    