import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch

def var_stand(directory):
    """
    :param directory: directory path to the netCDF datasets
     """

    ## LOAD THE DATASETS
    chloro = nc.Dataset(directory + 'chl_W-E-NS_month-yr_1960-2010_interp.nc', 'r', format = "NETCDF4")
    zoo = nc.Dataset(directory + 'cpr_10w10e_50n60n_month_1960-2010.nc', 'r', format = "NETCDF4")
    wind = nc.Dataset(directory + 'OAflux_v3_NAtl_1960-2009_wind-speed.nc', 'r', format = "NETCDF4")
    MLD = nc.Dataset(directory + 'SODA-v216_NAtl_mld_003-sigpot_1960-2010.nc', 'r', format = "NETCDF4")
    SST = nc.Dataset(directory + 'SODA-v216_NAtl_sst.nc', 'r', format = "NETCDF4")
    AMO = nc.Dataset(directory + 'AMO_Kaplan_noaa_Unsmooth_1960-2010.nc', 'r', format = "NETCDF4")
    NAO = nc.Dataset(directory + 'NAO-CPC_1960-2010.nc', 'r', format = "NETCDF4")

    ## TIME REFERENCES
    time = chloro['GTT'][6:-18]
    time_years = chloro['GTT'][6:-18]/12 + 1901
    dt = (time[1:] - time[:-1]).reshape(-1,1)

    ## MARK OUT THE SHIFT TIME FRAMES
    idx1980 = np.where(np.round(time_years,1) == 1980)[0][0]
    idx1981 = np.where(np.round(time_years,1) == 1981)[0][0]
    idx1989 = np.where(np.round(time_years,1) == 1989)[0][0]

    ## BIOLOGY
    chl = chloro['CHLN_INT'][6:-18]
    cop_NS = zoo['TOTCOP'][6:-18]

    ## PHYSICS : 1D VARIABLES
    AMO_NS = AMO['AMO'][6:-18]
    NAO_NS = NAO['INDEX'][7:-18]

    ## PHYSICS : 2D VARIABLES
    SST_NS  = []
    MLD_NS  = []
    wind_NS = []
    # Take the mean
    for i in range(len(SST['SST'][:])):
        SST_NS.append(np.mean(SST['SST'][i][0]))
        MLD_NS.append(np.mean(MLD['MLD'][i]))
        wind_NS.append(np.mean(wind['WND10'][i][0]))

    ## DATA STANDARDIZATION WITH P1 MEAN AND STD
    chloro_stand_P1 = (chl - np.mean(chl[:idx1980])) / np.std(chl[:idx1980])
    cop_stand_P1    = (cop_NS - np.mean(cop_NS[:idx1980])) / np.std(cop_NS[:idx1980])

    AMO_stand_P1    = (AMO_NS - np.mean(AMO_NS[:idx1980])) / np.std(AMO_NS[:idx1980])
    NAO_stand_P1    = (NAO_NS - np.mean(NAO_NS[:idx1980])) / np.std(NAO_NS[:idx1980])
    SST_stand_P1    = (SST_NS - np.mean(SST_NS[:idx1980])) / np.std(SST_NS[:idx1980])
    MLD_stand_P1    = (MLD_NS - np.mean(MLD_NS[:idx1980])) / np.std(MLD_NS[:idx1980])
    wind_stand_P1   = (wind_NS - np.mean(wind_NS[:idx1980])) / np.std(wind_NS[:idx1980])

    ## CONCATENATION
    co_var = np.column_stack((NAO_stand, MLD_stand))
    co_var = np.column_stack((co_var, SST_stand))
    co_var = np.column_stack((co_var, wind_stand))
    co_var = np.column_stack((co_var, AMO_stand))
    co_var = co_var.reshape(-1, 5)
    input = np.column_stack((chl_stand, co_var))

    inp_true_test = input[idx1981 + 1:].reshape(-1, input.shape[1])

    return time_years, inp_true_test, chloro_stand_P1, NAO_stand_P1, MLD_stand_P1, SST_stand_P1, wind_stand_P1, AMO_stand_P1


def inputs_rdn_1(time_years, chl_stand, NAO_stand, MLD_stand, SST_stand, wind_stand, AMO_stand, randomized_variable = "NAO"):

    ## MARK OUT THE SHIFT TIME FRAMES
    idx1980 = np.where(np.round(time_years, 1) == 1980)[0][0]
    idx1981 = np.where(np.round(time_years, 1) == 1981)[0][0]
    idx1989 = np.where(np.round(time_years, 1) == 1989)[0][0]
    dt = 12*(time_years[1:] - time_years[:-1]).reshape(-1, 1)

    random_variable = np.random.normal(0, 1, len(chl_stand))

    if randomized_variable == "NAO":
        NAO_stand = random_variable
    elif randomized_variable == "MLD":
        MLD_stand = random_variable
    elif randomized_variable == "SST":
        SST_stand = random_variable
    elif randomized_variable == "wind":
        wind_stand = random_variable
    elif randomized_variable == "AMO":
        AMO_stand = random_variable

    ## CONCATENATION
    co_var = np.column_stack((NAO_stand, MLD_stand))
    co_var = np.column_stack((co_var, SST_stand))
    co_var = np.column_stack((co_var, wind_stand))
    co_var = np.column_stack((co_var, AMO_stand))
    co_var = co_var.reshape(-1, 5)
    input = np.column_stack((chl_stand, co_var))

    ## TEST SPLIT
    inp_test = input[idx1981:-1].reshape(-1,input.shape[1])
    inp_true_test = input[idx1981 + 1:].reshape(-1,input.shape[1])
    dt_test = dt[idx1981:].reshape(-1,1)
    time_years_test = time_years[idx1981 + 1:]
    var_test = np.var(inp_test)

    return inp_test, inp_true_test, dt_test


def inputs_rdn_3(time_years, chl_stand, NAO_stand, MLD_stand, SST_stand, wind_stand, AMO_stand, dt_horizon, randomized_variable = 'NAO'):

    ## MARK OUT THE SHIFT TIME FRAMES
    idx1980 = np.where(np.round(time_years, 1) == 1980)[0][0]
    idx1981 = np.where(np.round(time_years, 1) == 1981)[0][0]
    idx1989 = np.where(np.round(time_years, 1) == 1989)[0][0]

    random_variable = np.random.normal(0, 1, len(chl_stand))

    if randomized_variable == "NAO":
        NAO_stand = random_variable
    elif randomized_variable == "MLD":
        MLD_stand = random_variable
    elif randomized_variable == "SST":
        SST_stand = random_variable
    elif randomized_variable == "wind":
        wind_stand = random_variable
    elif randomized_variable == "AMO":
        AMO_stand = random_variable

    ## CONCATENATION
    co_var = np.column_stack((NAO_stand, MLD_stand))
    co_var = np.column_stack((co_var, SST_stand))
    co_var = np.column_stack((co_var, wind_stand))
    co_var = np.column_stack((co_var, AMO_stand))
    co_var = co_var.reshape(-1, 5)
    input = np.column_stack((chl_stand, co_var))

    ## MODIFICATION OF THE DATASETS
    dt = 12 * (time_years[1:] - time_years[:-1]).reshape(-1, 1)
    dt_horizon = params['dt_horizon']

    ## TEST SPLIT
    inp_test = np.zeros([len(time_years_test) - 2 * dt_horizon, dt_horizon, input.shape[1]])
    inp_true_test = np.zeros([inp_test.shape[0], dt_horizon, input.shape[1]])
    for i in range(len(inp_test)):
        inp_test[i] = input[idx1981 + i: idx1981 + i + dt_horizon]
        inp_true_test[i] = input[idx1981 + i + dt_horizon: idx1981 + i + 2 * dt_horizon]
    var_test = np.var(inp_test[:, 0, 0])

    inp_test = torch.Tensor(inp_test).view(inp_test.shape[0], -1)
    inp_true_test = torch.Tensor(inp_true_test).view(inp_test.shape[0], -1)

    return inp_test, inp_true_test, dt_test
