import numpy as np 

def read_data(file_path):
    ###################################
    # each line is one hour
    # the fields are: pollution, dew, temp, pressure, wind_dir, wind_spd, snow, rain
    ###################################
    return np.load(file_path)