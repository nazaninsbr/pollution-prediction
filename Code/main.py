import data_processor
import constants
import model


def run_part_1_code(train_X, train_y, test_X, test_y):
    pass

def main():
    data = data_processor.read_data(constants.path_to_data)
    train_X, train_y, test_X, test_y = data_processor.split_and_prep_data(data, split_point = constants.test_train_split, window_size = constants.window_size)

    run_part_1_code(train_X, train_y, test_X, test_y)

main()