path_to_data = '../Data/polution_dataSet.npy'

window_size = 11
trian_split_part = 12000
test_split_part = 15001


loss_function = 'mean_squared_error'
optimizer = 'adam'
number_of_epochs = 10
batch_size = 32
validation_split = 0.1
add_dropout = False