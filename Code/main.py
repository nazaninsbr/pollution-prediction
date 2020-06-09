import data_processor
import constants
import model
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def run_window_11_codes_using_generator_data(train_X, train_y, test_X, test_y):
    # LSTM Model 
    lstm_1 = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q2_generator_lstm_model')
    lstm_1.test_different_loss_and_optimization_functions()

    lstm_1_with_dropout = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q5_generator_lstm_with_dropout_model')
    lstm_1_with_dropout.train_and_report_results()

    # GRU Model
    gru_1 = model.GRU_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q2_generator_GRU_model')
    gru_1.test_different_loss_and_optimization_functions()

    gru_1_with_dropout = model.GRU_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q5_generator_GRU_with_dropout_model')
    gru_1_with_dropout.train_and_report_results()

    # RNN Model
    rnn_1 = model.RNN_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q2_generator_RNN_model')
    rnn_1.test_different_loss_and_optimization_functions()

    rnn_1_with_dropout = model.RNN_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q5_generator_RNN_with_dropout_model')
    rnn_1_with_dropout.train_and_report_results()

def run_window_11_codes(train_X, train_y, test_X, test_y):
    # LSTM Model 
    lstm_1 = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q2_basic_lstm_model')
    lstm_1.test_different_loss_and_optimization_functions()

    lstm_1_with_dropout = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q5_lstm_with_dropout_model')
    lstm_1_with_dropout.train_and_report_results()

    # GRU Model
    gru_1 = model.GRU_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q2_basic_GRU_model')
    gru_1.test_different_loss_and_optimization_functions()

    gru_1_with_dropout = model.GRU_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q5_GRU_with_dropout_model')
    gru_1_with_dropout.train_and_report_results()

    # RNN Model
    rnn_1 = model.RNN_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q2_basic_RNN_model')
    rnn_1.test_different_loss_and_optimization_functions()

    rnn_1_with_dropout = model.RNN_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q5_RNN_with_dropout_model')
    rnn_1_with_dropout.train_and_report_results()

def run_part_4_codes(train_X, train_y, test_X, test_y, time_Series_type):
    # LSTM Model 
    lstm_1 = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/'+time_Series_type+'Q4_lstm_model')
    lstm_1.train_and_report_results()

    # GRU Model
    gru_1 = model.GRU_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/'+time_Series_type+'Q4_GRU_model')
    gru_1.train_and_report_results()

    # RNN Model
    rnn_1 = model.RNN_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/'+time_Series_type+'Q4_RNN_model')
    rnn_1.train_and_report_results()

def run_part_6_codes(train_X, train_y, test_X, test_y):
    q6_model = model.FusionModel(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q6_fusion_lstm_model')
    q6_model.train_fusion_model()
    
def run_part_7_code():
    numpy_file = np.load(constants.path_to_data)
    columns = [f'col_{num}' for num in range(8)]
    index = [f'index_{num}' for num in range(numpy_file.shape[0])]
    df = pd.DataFrame(numpy_file, columns=columns, index=index)
    df.columns = ['pollution', 'dew', 'temp', 'pressure', 'wind_dir', 'wind_spd', 'snow', 'rain']
    corr = df.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.show()

def part_one_codes(data):
    # using the data generator 
    train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size, method_name = 'generator')
    run_window_11_codes_using_generator_data(train_X, train_y, test_X, test_y)

    # reading the data and working with it in the window = 11 format
    # train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size, method_name = 'normal')
    # run_window_11_codes(train_X, train_y, test_X, test_y)

    # # answering part 4
    # train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size, method_name = 'weekly')
    # run_part_4_codes(train_X, train_y, test_X, test_y, 'weekly')

    # train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size, method_name = 'monthly')
    # run_part_4_codes(train_X, train_y, test_X, test_y, 'monthly')

    # # answering part 6
    # train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size, method_name = 'fusion')
    # run_part_6_codes(train_X, train_y, test_X, test_y)

    # answering part 7
    # run_part_7_code()

def main():
    data = data_processor.read_data(constants.path_to_data)
    part_one_codes(data)


warnings.filterwarnings("ignore")
main()