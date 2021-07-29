## IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

def time_serie_all(directory, MLP_outputs_test, MLP_outputs_train, Chl_stand_test, Chl_stand_train, pred_test_style,
                   pred_train_style, Chl_style, graph_settings, model=1):
    """
    :param directory: path to the Chl CPR dataset
    :param MLP_outputs_test: model's predictions using the first element of test as the initial condition
    :param MLP_outputs_train: model's predictions using the first element of train as the initial condition
    :param Chl_stand_test: target values of test
    :param Chl_stand_train: target values of train
    :param pred_test_style: a dictionnary with keys ['color', 'label', 'linewidth', 'linestyle', 'mean_color', 'mean_label', 'mean_linewidth', 'mean_linestyle', 'marker'] for MLP predicitions on test
    :param pred_train_style: a dictionnary with keys ['color', 'label', 'linewidth', 'linestyle', 'mean_color', 'mean_label', 'mean_linewidth', 'mean_linestyle', 'marker'] for MLP predicitions on train
    :param Chl_style: a dictionnary with keys ['color', 'label', 'linewidth', 'linestyle', 'mean_all_color', 'mean_all_label', 'mean_all_linewidth', 'mean_all_linestyle', 'mean_test_color', 'mean_test_label', 'mean_test_linewidth', 'mean_test_linestyle', 'mean_train_color', 'mean_train_label', 'mean_train_linewidth', 'mean_train_linestyle', 'marker'] for Chl CPR
    :param graph_settings: a dictionnary with keys ['figsize', 'xlabel', 'ylabel', 'title', 'title_size', 'legend_size']
    :param model: the number of the method used for reconstructio : {1 ; 2 ; 3}
    """

    chloro = nc.Dataset(directory + 'chl_W-E-NS_month-yr_1960-2010_interp.nc', 'r', format="NETCDF4")
    chl = chloro['CHLN_INT'][6:-18]

    ## CALCULATE THE ANOMALIES (KNOWING THAT THE DATA IS STANDARDIZED WITH P1 MEAN AND STD)
    anomaly_all = chl - np.mean(chl)
    anomaly_train = (Chl_stand_train) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)
    anomaly_test = (Chl_stand_test) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)
    anomaly_pred_train = np.array(MLP_outputs_train) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)
    anomaly_pred_test = np.array(MLP_outputs_test) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)

    ## PLOTS
    plt.figure(figsize=graph_settings['figsize'])
    plt.plot(time_years, anomaly_all, color=Chl_style['color'], label=Chl_style['label'],linewidth=Chl_style['linewidth'], linestyle=Chl_style['linestyle'], marker=Chl_style['marker'])
    if model == 1:
        plt.plot(time_years_test, anomaly_pred_test, color=pred_test_style['color'], label=pred_test_style['label'],linewidth=pred_test_style['linewidth'], linestyle=pred_test_style['linestyle'], marker=pred_test_style['marker'])
        plt.plot(time_years_train, anomaly_pred_train, color=pred_train_style['color'], label=pred_train_style['label'],linewidth=pred_train_style['linewidth'], linestyle=pred_train_style['linestyle'], marker=pred_test_style['marker'])
    elif model == 2:
        plt.plot(time_years_test[1:-params['dt_horizon']], anomaly_pred_test, color=pred_test_style['color'],label=pred_test_style['label'], linewidth=pred_test_style['linewidth'], linestyle=pred_test_style['linestyle'], marker=pred_test_style['marker'])
        plt.plot(time_years_train[1:-params['dt_horizon']], anomaly_pred_train, color=pred_train_style['color'],label=pred_train_style['label'], linewidth=pred_train_style['linewidth'], linestyle=pred_train_style['linestyle'], marker=pred_test_style['marker'])
    else :
        plt.plot(time_years_test[params['dt_horizon']:-params['dt_horizon']], anomaly_pred_test, color=pred_test_style['color'], label=pred_test_style['label'], linewidth=pred_test_style['linewidth'], linestyle=pred_test_style['linestyle'], marker=pred_test_style['marker'])
        plt.plot(time_years_train[params['dt_horizon']:-params['dt_horizon']], anomaly_pred_train, color=pred_train_style['color'], label=pred_train_style['label'], linewidth=pred_train_style['linewidth'], linestyle=pred_train_style['linestyle'], marker=pred_test_style['marker'])
    plt.plot(time_years, np.zeros(len(time_years)), color=Chl_style['mean_all_color'],label=Chl_style['mean_all_label'], linestyle=Chl_style['mean_all_linestyle'], linewidth=Chl_style['mean_all_linewidth'])
    plt.plot(time_years_train, np.mean(anomaly_train) * np.ones(len(time_years_train)),color=Chl_style['mean_train_color'], label=Chl_style['mean_train_label'], linestyle=Chl_style['mean_train_linestyle'], linewidth=Chl_style['mean_train_linewidth'])
    plt.plot(time_years_train, np.mean(anomaly_pred_train) * np.ones(len(time_years_train)),color=pred_train_style['mean_color'], label=pred_train_style['mean_label'], linestyle=pred_train_style['mean_linestyle'], linewidth=pred_train_style['mean_linewidth'])
    plt.plot(time_years_test, np.mean(anomaly_pred_test) * np.ones(len(time_years_test)), color=pred_test_style['mean_color'], label=pred_test_style['mean_label'], linestyle=pred_test_style['mean_linestyle'], linewidth=pred_test_style['mean_linewidth'])
    plt.plot(time_years_test, np.mean(anomaly_test) * np.ones(len(time_years_test)), color=Chl_style['mean_test_color'], label=Chl_style['mean_test_label'], linestyle=Chl_style['mean_test_linestyle'], linewidth=Chl_style['mean_test_linewidth'])
    plt.xlabel(graph_settings['xlabel'])
    plt.ylabel(graph_settings['ylabel'])
    plt.title(graph_settings['title'], size=graph_settings['title_size'], fontweight='bold')
    plt.legend(prop={'size': graph_settings['legend_size']})


def time_serie_test(directory, MLP_outputs_test, Chl_stand_test, Chl_stand_train, pred_test_style, Chl_style, graph_settings, model=1):
    """
    :param directory: path to the Chl CPR dataset
    :param MLP_outputs_test: model's predictions using the first element of test as the initial condition
    :param Chl_stand_test: target values of test
    :param Chl_stand_train: target values of train
    :param pred_test_style: a dictionnary with keys ['color', 'label', 'linewidth', 'linestyle', 'mean_color', 'mean_label', 'mean_linewidth', 'mean_linestyle', 'marker'] for MLP predicitions on test
    :param Chl_style: a dictionnary with keys ['color', 'label', 'linewidth', 'linestyle', 'mean_all_color', 'mean_all_label', 'mean_all_linewidth', 'mean_all_linestyle', 'mean_test_color', 'mean_test_label', 'mean_test_linewidth', 'mean_test_linestyle', 'mean_train_color', 'mean_train_label', 'mean_train_linewidth', 'mean_train_linestyle', 'marker', 'test_label'] for Chl CPR
    :param graph_settings: a dictionnary with keys [''figsize', 'xlabel', 'ylabel', 'title', 'title_size', 'legend_size']
    :param model: the number of the method used for reconstructio : {1 ; 2 ; 3}
    """
    chloro = nc.Dataset(directory + 'chl_W-E-NS_month-yr_1960-2010_interp.nc', 'r', format="NETCDF4")
    chl = chloro['CHLN_INT'][6:-18]

    anomaly_test = (Chl_stand_test) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)
    anomaly_pred_test = np.array(MLP_outputs_test) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)
    anomaly_train = (Chl_stand_train) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)

    plt.figure(figsize=(20, 4))
    if model == 1:
        plt.plot(time_years_test, anomaly_test, label=Chl_style['test_label'], color=Chl_style['color'], linestyle=Chl_style['linestyle'], linewidth=Chl_style['linewidth'], marker=Chl_style['marker'])
        plt.plot(time_years_test, anomaly_pred_test, label=pred_test_style['label'], color=pred_test_style['color'], linestyle=pred_test_style['linestyle'], linewidth=pred_test_style['linewidth'], marker=pred_test_style['marker'])
    elif model == 2:
        plt.plot(time_years_test[1:-params['dt_horizon']], anomaly_test, label=Chl_style['test_label'], color=Chl_style['color'],linestyle=Chl_style['linestyle'], linewidth=Chl_style['linewidth'],marker=Chl_style['marker'])
        plt.plot(time_years_test[1:-params['dt_horizon']], anomaly_pred_test, label=pred_test_style['label'], color=pred_test_style['color'],linestyle=pred_test_style['linestyle'], linewidth=pred_test_style['linewidth'],marker=pred_test_style['marker'])
    else:
        plt.plot(time_years_test[params['dt_horizon']:-params['dt_horizon']], anomaly_test, label=Chl_style['test_label'], color=Chl_style['color'], linestyle=Chl_style['linestyle'], linewidth=Chl_style['linewidth'], marker=Chl_style['marker'])
        plt.plot(time_years_test[params['dt_horizon']:-params['dt_horizon']], anomaly_pred_test, label=pred_test_style['label'], color=pred_test_style['color'], linestyle=pred_test_style['linestyle'], linewidth=pred_test_style['linewidth'], marker=pred_test_style['marker'])
    plt.plot(time_years_test, np.mean(anomaly_test) * np.ones(len(time_years_test)), color=Chl_style['mean_test_color'],label=Chl_style['mean_test_label'], linestyle=Chl_style['mean_test_linestyle'],linewidth=Chl_style['mean_test_linewidth'])
    plt.plot(time_years_test, np.mean(anomaly_pred_test) * np.ones(len(time_years_test)),color=pred_test_style['mean_color'], label=pred_test_style['mean_label'],linestyle=pred_test_style['mean_linestyle'], linewidth=pred_test_style['mean_linewidth'])
    plt.plot(time_years_test, np.zeros(len(time_years_test)), color=Chl_style['mean_all_color'], label=Chl_style['mean_all_label'], linestyle=Chl_style['mean_all_linestyle'], linewidth=Chl_style['mean_all_linewidth'])
    plt.plot(time_years_test, np.mean(anomaly_train) * np.ones(len(time_years_test)),color=Chl_style['mean_train_color'], label=Chl_style['mean_train_label'],linestyle=Chl_style['mean_train_linestyle'], linewidth=Chl_style['mean_train_linewidth'])
    plt.xlabel(graph_settings['xlabel'])
    plt.ylabel(graph_settings['ylabel'])
    plt.title(graph_settings['title'], size=graph_settings['title_size'], fontweight='bold')
    plt.legend(prop={'size': graph_settings['legend_size']})

def annualMeans(directory, MLP_outputs_test, MLP_outputs_train, Chl_stand_test, Chl_stand_train):
    """
    :param directory: path to the Chl CPR dataset
    :param MLP_outputs_test: model's predictions using the first element of test as the initial condition
    :param MLP_outputs_train: model's predictions using the first element of train as the initial condition
    :param Chl_stand_test: target values of test
    :param Chl_stand_train: target values of train
    """
    chloro = nc.Dataset(directory + 'chl_W-E-NS_month-yr_1960-2010_interp.nc', 'r', format="NETCDF4")
    chl = chloro['CHLN_INT'][6:-18]

    ## CALCULATE THE ANOMALIES (KNOWING THAT THE DATA IS STANDARDIZED WITH P1 MEAN AND STD)
    anomaly_all = chl - np.mean(chl)
    anomaly_pred_train = np.array(MLP_outputs_train) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean( chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)
    anomaly_pred_test = np.array(MLP_outputs_test) * np.std(chl[:idx1980 - params['dt_horizon']]) + np.mean(chl[:idx1980 - params['dt_horizon']]) - np.mean(chl)

    year_mean_train = []
    cumulated_sum_train = [0]
    sum = 0
    for i in range(0, len(anomaly_pred_train), 12):
        year_mean_train.append(np.mean(anomaly_pred_train[i:i + 12]))
        sum += year_mean_train[-1]
        cumulated_sum_train.append(sum)

    year_mean_test = []
    cumulated_sum_test = [0]
    sum = 0
    for i in range(0, len(anomaly_pred_test), 12):
        year_mean_test.append(np.mean(anomaly_pred_test[i:i + 12]))
        sum += year_mean_test[-1]
        cumulated_sum_test.append(sum)

    year_mean = []
    cumulated_sum = [0]
    sum = 0
    for i in range(0, len(anomaly_all), 12):
        year_mean.append(np.mean(anomaly_all[i:i + 12]))
        sum += year_mean[-1]
        cumulated_sum.append(sum)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(np.arange(1961, 2010, 1), year_mean, color='black', label='real')
    if params['dt_horizon'] == 0:
        ax.plot(np.arange(1961, 1981, 1), year_mean_train, color='black', label='train (P1)', linestyle='dashed')
        ax.plot(np.arange(1981, 2010, 1), year_mean_test, color='black', label='test (P2)', linestyle='dotted')
    else :
        ax.plot(np.arange(1961.5, 1980.5, 1), year_mean_train, color='black', label='train (P1)', linestyle='dashed')
        ax.plot(np.arange(1981.5, 2009.5, 1), year_mean_test, color='black', label='test (P2)', linestyle='dotted')
    ax.set_ylabel('Annual means of Chl anomalies', size=13)
    ax.set_xlabel('Time (years)', size=13)
    plt.legend(prop={'size': 13}, loc='upper left')
    ax2 = ax.twinx()
    if params['dt_horizon'] == 0:
        ax2.plot(np.arange(1981, 2010, 1), cumulated_sum_test[:-1], color='red', linestyle='dotted')
        ax2.plot(np.arange(1961, 1981, 1), cumulated_sum_train[:-1], color='red', linestyle='dashed')
    else :
        ax2.plot(np.arange(1981.5, 2009.5, 1), cumulated_sum_test[:-1], color='red', linestyle='dotted')
        ax2.plot(np.arange(1961.5, 1980.5, 1), cumulated_sum_train[:-1], color='red', linestyle='dashed')
    ax2.plot(np.arange(1961, 2010, 1), cumulated_sum[:-1], color='red')
    ax2.set_ylabel("Cumulated sums of Chl anomalies", size=13)
    ax2.spines['right'].set_color('red')
    ax2.yaxis.label.set_color("red")
    ax2.tick_params(axis='y', colors='red')
    plt.title("Annual means (black) and their cumulated sums (red), of Chl CPR and Chl MLP on P1 and P2.", size=20,fontweight='bold')


def scatterplots(MLP_outputs_test, MLP_outputs_train, Chl_stand_test, Chl_stand_train):
    """
    :param MLP_outputs_test: model's predictions using the first element of test as the initial condition
    :param MLP_outputs_train: model's predictions using the first element of train as the initial condition
    :param Chl_stand_test: target values of test
    :param Chl_stand_train: target values of train
    """
    ## SCATTER PLOT ON BOTH TRAIN AND TEST
    MLP_outputs_all = np.concatenate((MLP_outputs_train, MLP_outputs_test), axis=None).reshape(-1,1)
    Chl_stand_all = np.concatenate((Chl_stand_train, Chl_stand_test), axis=None).reshape(-1,1)
    linReg = LinearRegression(fit_intercept=False)
    linReg.fit(MLP_outputs_all, Chl_stand_all)
    a = linReg.coef_
    R2 = linReg.score(MLP_outputs_all, Chl_stand_all)

    plt.figure(figsize=(8, 7))
    plt.scatter(np.array(MLP_outputs_train).reshape(1, -1), np.array(Chl_stand_train).reshape(1, -1), c='dodgerblue',alpha=0.5, label='train')
    plt.scatter(np.array(MLP_outputs_test).reshape(1, -1), np.array(Chl_stand_test).reshape(1, -1), c='deeppink',alpha=0.5, label='test')
    plt.plot(np.arange(-1.5, 4.5, 0.5), np.arange(-1.5, 4.5, 0.5), color='red')
    plt.plot(np.arange(-1.5, 4.5, 0.5), a[0] * np.arange(-1.5, 4.5, 0.5), color='black',label='slope = ' + str(round(float(a[0]), 3)) + '\n' + 'R² = ' + str(round(R2, 3)))
    plt.legend(prop={'size': 13})
    plt.xlabel('Chl MLP', size=11)
    plt.ylabel('Chl CPR', size=11)
    plt.title("Regression MLP on P1+P2* (black) vs. Regression Line 1:1 (red)", fontweight='bold', size=13)

    ## SCATTER PLOT JUST ON TRAIN
    linReg1 = LinearRegression(fit_intercept=False)
    linReg1.fit(np.array(MLP_outputs_train).reshape(-1, 1), np.array(Chl_stand_train).reshape(-1, 1))
    a_1 = linReg1.coef_
    R2_1 = linReg1.score(np.array(MLP_outputs_train).reshape(-1, 1), np.array(Chl_stand_train).reshape(-1, 1))

    f = plt.figure(figsize=(24, 8))

    xy_train = np.vstack([MLP_outputs_train, Chl_stand_train])
    z_train = gaussian_kde(xy_train)(xy_train)

    ax1 = f.add_subplot(121)
    scatt_1 = ax1.scatter(np.array(MLP_outputs_train).reshape(-1, 1), np.array(Chl_stand_train).reshape(-1, 1),c=z_train, alpha=0.7)
    ax1.plot(np.arange(-2, 2.5, 0.5), np.arange(-2, 2.5, 0.5), color='red')
    ax1.plot(np.arange(-2, 2.5, 0.5), a_1[0] * np.arange(-2, 2.5, 0.5), color='black',label='slope = ' + str(round(float(a_1[0]), 3)) + '\n' + 'R² = ' + str(round(R2_1, 3)))
    f.colorbar(mappable=scatt_1).set_label("Data distribution density", fontweight='bold', labelpad=20, size =14)
    plt.xticks(np.arange(-2, 2.5, 1))
    plt.yticks(np.arange(-2, 2.5, 1))
    plt.legend(prop={'size': 13})
    plt.xlabel('Chl MLP', size=14)
    plt.ylabel('Chl CPR', size=14)
    plt.title("Regression MLP on P1 (black) vs. Regression Line 1:1 (red)", fontweight='bold', size=14)

    linReg2 = LinearRegression(fit_intercept=False)
    linReg2.fit(np.array(MLP_outputs_test).reshape(-1, 1), np.array(Chl_stand_test).reshape(-1, 1))
    a_2 = linReg2.coef_
    R2_2 = linReg2.score(np.array(MLP_outputs_test).reshape(-1, 1), np.array(Chl_stand_test).reshape(-1, 1))

    xy_test = np.vstack([MLP_outputs_test, Chl_stand_test])
    z_test = gaussian_kde(xy_test)(xy_test)

    ax2 = f.add_subplot(122)
    scatt_2 = ax2.scatter(np.array(MLP_outputs_test).reshape(-1, 1), np.array(Chl_stand_test).reshape(-1, 1),c=z_test, alpha=0.7)
    ax2.plot(np.arange(-2, 5, 0.5), np.arange(-2, 5, 0.5), color='red')
    ax2.plot(np.arange(-2, 5, 0.5), a_2[0] * np.arange(-2, 5, 0.5), color='black',label='slope = ' + str(round(float(a_2[0]), 3)) + '\n' + 'R² = ' + str(round(R2_2, 3)))
    f.colorbar(mappable=scatt_2).set_label("Data distribution density", fontweight='bold', labelpad=20, size=14)
    plt.xticks(np.arange(-2, 5, 1))
    plt.yticks(np.arange(-2, 5, 1))
    plt.legend(prop={'size': 13})
    plt.xlabel('Chl MLP', size=14)
    plt.ylabel('Chl CPR', size=14)
    plt.title("Regression MLP on P2 (black) vs. Regression Line 1:1 (red)", fontweight='bold', size=14)