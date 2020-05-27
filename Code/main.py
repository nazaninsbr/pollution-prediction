import data_processor
import constants
import model


def run_part_1_code(train_X, train_y, test_X, test_y):
    # LSTM Model 
    lstm_1 = model.LSTM_Model(train_X, train_y, test_X, test_y, 
                loss_function = constants.loss_function, optimizer = constants.optimizer, validation_split = constants.validation_split,
                number_of_epochs = constants.number_of_epochs, batch_size = constants.batch_size, 
                file_save_name = '../Generated-Files/basic_lstm_model')
    lstm_1.train_and_report_results()

def main():
    data = data_processor.read_data(constants.path_to_data)
    train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = (constants.trian_split_part, constants.test_split_part), window_size = constants.window_size)

    run_part_1_code(train_X, train_y, test_X, test_y)

main()