import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch

def inputs_3(directory, dt_horizon):
    """
    :param directory: directory path to the netCDF datasets
    :param dt_horizon: the number of previous states to take into account and the number of states to forecast
    :return: all train and test vectors (neural net input, input target, time step) to put in the dataloaders, as well as time scales.
    """

    ## LOAD THE DATASETS
    chloro = nc.Dataset(directory + 'chl_W-E-NS_month-yr_1960-2010_interp.nc', 'r', format="NETCDF4")
    zoo = nc.Dataset(directory + 'cpr_10w10e_50n60n_month_1960-2010.nc', 'r', format="NETCDF4")
    wind = nc.Dataset(directory + 'OAflux_v3_NAtl_1960-2009_wind-speed.nc', 'r', format="NETCDF4")
    MLD = nc.Dataset(directory + 'SODA-v216_NAtl_mld_003-sigpot_1960-2010.nc', 'r', format="NETCDF4")
    SST = nc.Dataset(directory + 'SODA-v216_NAtl_sst.nc', 'r', format="NETCDF4")
    AMO = nc.Dataset(directory + 'AMO_Kaplan_noaa_Unsmooth_1960-2010.nc', 'r', format="NETCDF4")
    NAO = nc.Dataset(directory + 'NAO-CPC_1960-2010.nc', 'r', format="NETCDF4")

    ## TIME REFERENCES
    time = chloro['GTT'][6:-18]
    time_years = chloro['GTT'][6:-18] / 12 + 1901
    dt = (time[1:] - time[:-1]).reshape(-1, 1)

    ## MARK OUT THE SHIFT TIME FRAMES
    idx1980 = np.where(np.round(time_years, 1) == 1980)[0][0]
    idx1981 = np.where(np.round(time_years, 1) == 1981)[0][0]
    idx1989 = np.where(np.round(time_years, 1) == 1989)[0][0]
    print("P1    = [1960;1980]      <=> Indices = [0;%d]" % idx1980)
    print("P2    = [1981;2010]      <=> Indices = [%d;%d]" % (idx1981, len(time_years) - 1))
    print("SHIFT = [1981;1989] ⊂ P2 <=> Indices = [%d;%d]" % (idx1981, idx1989))
    print()
    ## BIOLOGY
    chl = chloro['CHLN_INT'][6:-18]
    cop_NS = zoo['TOTCOP'][6:-18]

    ## PHYSICS : 1D VARIABLES
    AMO_NS = AMO['AMO'][6:-18]
    NAO_NS = NAO['INDEX'][7:-18]

    ## PHYSICS : 2D VARIABLES
    SST_NS = []
    MLD_NS = []
    wind_NS = []
    # Take the mean
    for i in range(len(SST['SST'][:])):
        SST_NS.append(np.mean(SST['SST'][i][0]))
        MLD_NS.append(np.mean(MLD['MLD'][i]))
        wind_NS.append(np.mean(wind['WND10'][i][0]))

    ## DATA STANDARDIZATION WITH P1 MEAN AND STD
    chloro_stand_P1 = (chl - np.mean(chl[:idx1980])) / np.std(chl[:idx1980])
    cop_stand_P1 = (cop_NS - np.mean(cop_NS[:idx1980])) / np.std(cop_NS[:idx1980])

    AMO_stand_P1 = (AMO_NS - np.mean(AMO_NS[:idx1980])) / np.std(AMO_NS[:idx1980])
    NAO_stand_P1 = (NAO_NS - np.mean(NAO_NS[:idx1980])) / np.std(NAO_NS[:idx1980])
    SST_stand_P1 = (SST_NS - np.mean(SST_NS[:idx1980])) / np.std(SST_NS[:idx1980])
    MLD_stand_P1 = (MLD_NS - np.mean(MLD_NS[:idx1980])) / np.std(MLD_NS[:idx1980])
    wind_stand_P1 = (wind_NS - np.mean(wind_NS[:idx1980])) / np.std(wind_NS[:idx1980])

    ## CONCATENATION
    co_var = np.column_stack((NAO_stand_P1, MLD_stand_P1))
    co_var = np.column_stack((co_var, SST_stand_P1))
    co_var = np.column_stack((co_var, wind_stand_P1))
    co_var = np.column_stack((co_var, AMO_stand_P1))
    co_var = co_var.reshape(-1, 5)
    input = np.column_stack((chloro_stand_P1, co_var))
    print("Input vector shape :", input.shape)
    print()

    ## MODIFICATION OF THE DATASETS
    dt = (time[1:] - time[:-1]).reshape(-1, 1)
    dt_horizon = params['dt_horizon']

    ## TRAIN SPLIT
    plt.figure(figsize=(20, 4))
    time_years_train = time_years[:idx1980 + 1]
    inp_train = np.zeros([len(time_years_train) - 2 * dt_horizon, dt_horizon, input.shape[1]])
    inp_true_train = np.zeros([inp_train.shape[0], dt_horizon, input.shape[1]])
    for i in range(len(inp_train)):
        inp_train[i] = input[i: i + dt_horizon]
        inp_true_train[i] = input[i + dt_horizon: i + 2 * dt_horizon]
    var_train = np.var(inp_train[:, 0, 0])

    plt.plot(time_years_train[:-2 * dt_horizon], inp_train[:, 0, 0], color='dodgerblue',
             label='Chl CPR,stand on P1 (train)')
    mean_train = np.mean(inp_train[:, 0, 0])

    time_years_train[:-dt_horizon]
    inp_train = torch.Tensor(inp_train).view(inp_train.shape[0], -1)
    inp_true_train = torch.Tensor(inp_true_train).view(inp_train.shape[0], -1)

    print("Input  train vector of shape :", inp_train.shape, " and variance :", var_train)
    print("Target train vector of shape :", inp_true_train.shape)
    print()
    time_years_test = time_years[idx1981:]
    inp_test = np.zeros([len(time_years_test) - 2 * dt_horizon, dt_horizon, input.shape[1]])
    inp_true_test = np.zeros([inp_test.shape[0], dt_horizon, input.shape[1]])
    for i in range(len(inp_test)):
        inp_test[i] = input[idx1981 + i: idx1981 + i + dt_horizon]
        inp_true_test[i] = input[idx1981 + i + dt_horizon: idx1981 + i + 2 * dt_horizon]
    var_test = np.var(inp_test[:, 0, 0])

    plt.plot(time_years_test[:-2 * dt_horizon], inp_test[:, 0, 0], color='deeppink', label='Chl CPR,stand on P2 (test)')
    mean_test = np.mean(inp_test[:, 0, 0])

    time_years_test[:-dt_horizon]
    inp_test = torch.Tensor(inp_test).view(inp_test.shape[0], -1)
    inp_true_test = torch.Tensor(inp_true_test).view(inp_test.shape[0], -1)

    print("Input  test  vector of shape :", inp_test.shape, " and variance :", var_test)
    print("Target test  vector of shape :", inp_true_test.shape)

    plt.plot(time_years, mean_train * np.ones_like(time_years), color='dimgrey', linestyle='dashed', label='Mean P1')
    plt.plot(time_years_test, mean_test * np.ones_like(time_years_test), color='darkviolet', label='Mean P2')
    plt.legend(prop={'size': 13}, loc='upper left')
    plt.title('Train and test sets.', fontweight='bold', size=20)

    return time_years, inp_train, inp_true_train, time_years_train, var_train, inp_test, inp_true_test, time_years_test, var_test
