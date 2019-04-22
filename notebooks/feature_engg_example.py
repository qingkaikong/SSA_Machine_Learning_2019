#*********************************************;
#  Feature Engineering Example using TSFRESH  ;
#  AUTHORS: Maruti Mudunuru, EES-16, LANL     ;
#  DATE MODIFIED: April-21-2017               ;
#*********************************************;

import numpy as np
import time
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
#
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction \
     import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters

#=========================;
#  Start processing time  ;
#=========================;
tic = time.process_time()

#=============================;
#  Function-1: plot a signal  ;
#=============================;
def plot_each_signal(files_npy, str_label_x, str_label_y, \
                     str_plot_title, color_data, start, end, i):

    #--------------------;
    #  Plot each signal  ;
    #--------------------;
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    #
    fl = files_npy[i] #.npy file name
    #print(fl)
    plt.rc('text', usetex = True)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_label_x, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_label_y, fontsize = 24, fontweight = 'bold')
    #plt.title(str_plot_title, fontsize = 24, fontweight = 'bold')
    plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    col  = color_data
    data = np.load(fl) #Load .npy file
    t    = 1e6 * data[:,0] #time
    y1   = data[:,1] - np.mean(data[:,1]) #Mean-shift of sensor-pair-1
    y2   = data[:,2] - np.mean(data[:,2]) #Mean-shift of sensor-pair-2
    ax.plot(t[start:end], y1[start:end], linestyle = 'solid', linewidth = 0.00005, color = col) #Sensor-pair-1 plot
    #ax.set_xlim([0, 100])
    #ax.set_ylim([-1.2, 1.2])
    #tick_spacing = 1
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fig.tight_layout()
    str_fig_name = 'data' + str(i)
    fig.savefig(str_fig_name + '.pdf')
    #fig.savefig(str_fig_name + '.eps', format = 'eps', dpi = 1000)
    #fig.savefig(str_fig_name + '.png')
    #plt.show()

#==============================================================;
#  Function-2: Put signal data in pandas dataframe so that it  ;
#            can be fed to TSFRESH package                     ;
#            Format = (#id, #time, #Amplitude)                 ;
#==============================================================;
def create_id_time_amplitude_matrix(id_val, fl, start, end):

    #----------------------------------------;
    #  Build id-time-timeSeriesValue matrix  ;
    #----------------------------------------;
    t_data_matrix = np.array([]) #id-time-timeSeriesValue
    data          = np.load(fl) #Load .npy file
    t             = data[start:end,0] #First column data is time
    num_t,        = t.shape #Total number of time-series data points
    t_int_list    = list(range(0,num_t)) #Column = 'time' list
    id_list       = id_val * np.ones(num_t, dtype = int) #Create id_list (column = 'id')
    t_data_matrix = np.append(t_data_matrix, id_list) #Append id_list
    t_data_matrix = np.stack((t_data_matrix, t_int_list), axis = -1) #Append time list
    y1            = np.asarray([data[start:end,1]]) #Sensor Pair-1 data
    t_data_matrix = np.concatenate((t_data_matrix, y1.T), axis = 1) #Format as follows (#id, #time, #Amplitude) 

    return t_data_matrix

#=====================================================================;
#  Function-3: Concatenate individual id-time-timeSeriesValue matrix  ;
#              and form pandas dataframe for feature engineering      ;
#=====================================================================;
def create_pd_dataframe(num_files, id_list, files_npy, start, end):

    #---------------------------------------------------------;
    #  Concatenate different matrices to form full ts-matrix  ;
    #---------------------------------------------------------;
    for i in range(0,num_files):
        id_val        = id_list[i]
        fl            = files_npy[i]
        t_data_matrix = create_id_time_amplitude_matrix(id_val, fl, start, end) #Create (#id, #time, #Amplitude) for a given .npy file and id_val
        #
        if i == 0:
            ts_matrix = copy.deepcopy(t_data_matrix)
        else:
            ts_matrix = np.concatenate((ts_matrix, t_data_matrix))

    #------------------------------------------------;
    #  Create PANDAS dataframe for entire ts-matrix  ;
    #------------------------------------------------;
    df      = pd.DataFrame(ts_matrix) #Create pandas dataframe for full ts_matrix
    slc     = np.r_[0,1,2]
    df[slc[0:2]] = df[slc[0:2]].astype(float) #Convert id and time to integers
    df[slc[2]] = df[slc[2]].astype(float) #Convert id and time to integers

    return df

#*********************************************************;
#  INPUTS: For feature engineering example using TSFRESH  ;
#*********************************************************;
settings       = EfficientFCParameters() #Low-cost computational features = 100
num_processors = 0 #Using multiprocessing module for feature extraction in TSFRESH
#
files_npy      = glob.glob('../data/ftrs_extraction/*.npy') #read .npy data files
num_files,     = np.shape(files_npy) #Total number of .npy files
#
start          = 0 #Start of time-seires
end            = 10000 #Size of each .npy file (End of time-series)
#
color_data     = 'b'
str_label_x    = r'Time [$10^{-6}$ s]'
str_label_y    = r'Amplitude [V]'
str_plot_title = r'Acoustic signal at gas volume fraction = $0 \%$'
#
id_list        = np.arange(num_files, dtype = int) #Create id_list
#
show_signal    = False

#******************************************;
#  Step-1: Plot signals for visualization  ;
#******************************************;
if show_signal:
    for i in range(0,num_files):
        plot_each_signal(files_npy, str_label_x, str_label_y, str_plot_title, color_data, start, end, i)

#***************************************************;
#  Step-2: Get data that is TSFRESH readble format  ;
#          (which is in a pandas DataFrame)         ;
#***************************************************;
num_files = 3
start     = 0
end       = 1000
df = create_pd_dataframe(num_files, id_list, files_npy, start, end)

#******************************************;
#  Step-3: Extract features using TSFRESH  ;
#******************************************;
settings       = EfficientFCParameters()
num_processors = 0
extracted_features  = extract_features(df, column_id = 0, column_sort = 1, \
                                       default_fc_parameters = settings, \
                                       n_jobs = num_processors, profile = False)
print(extracted_features)
#extracted_features.to_csv('Extracted_Features.csv')

#Note -- There seems to be an issue with pandas verions 0.24.2 etc. So install or downgrade to version 0.23.4
#https://github.com/blue-yonder/tsfresh/issues/485

#======================;
# End processing time  ;
#======================;
toc = time.process_time()
print('Time elapsed in seconds = ', toc - tic)
