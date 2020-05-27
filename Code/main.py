import data_processor
import constants
import model
import warnings


def run_Q1_code(train_X, train_y, test_X, test_y):
    # LSTM Model 
    lstm_1 = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = constants.add_dropout,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q1_basic_lstm_model')
    lstm_1.test_different_loss_and_optimization_functions()

    lstm_1_with_dropout = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split, add_dropout = True,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/Q1_lstm_with_dropout_model')
    lstm_1_with_dropout.train_and_report_results()

    # GRU Model

def run_code_for_initial_instructed_window_size(data):
    train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size, method_name = 'normal')
    run_Q1_code(train_X, train_y, test_X, test_y)

def main():
    data = data_processor.read_data(constants.path_to_data)
    run_code_for_initial_instructed_window_size(data)


warnings.filterwarnings("ignore")
main()