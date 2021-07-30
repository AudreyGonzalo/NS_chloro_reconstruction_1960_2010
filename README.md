# NS_chloro_reconstruction_1960_2010
 
## Introduction

### Motivation
Phytoplankton plays a key role in the regulation of oceanic and atmospheric carbon dioxide. Consequently, understanding the mecanisms of its evolution over time, through its interactions with the environment, is a core issue to forecast the long-term climate impact. 

This repository adresses the reconstruction of the North Sea chlorophyll (a phytoplankton biomass proxy) from 1960 to 2010, resorting to Deep Learning. More precisely, the corresponding dataset highlights a **pytoplankton regime shift in the mid 1980's**, as sketched below (Martinez et al., 2015):

![shift](/figures/Martinez_et_al_shift.PNG)

More specifically, the challenge tackled with this dataset is to train an AI to **predict the biological shift**, by detecting a long-term change in the phyics as of the 1980's. Concretely, it boils down to training the neural network only on the period prior to the shift (P1), the rest being the test set (P2). The two sets are considered as independant by imposing a ceasure of one year between them. 

![shift](/figures/timeline.PNG)

### Repository structure
This repository is oragnized as follows : 
* In the `data` folder, you may find all the datasets, with format `netCDF`.
* The `notebooks` folder gathers :
    * A brief vizualization of the data, and its formatting process.
    * The tested approaches (or methods) from `baseline`.
    * Complementary results such as :_
        * The assessment of the variables' strenght in each of the `baseline` models.
        * The impact of zooplankton when it is added to the physical predictors.  
* The `models` folder, with the weights of each different type of model saved from their respective notebook in `basline`.
* The `src` folder contains all the python codes for data formating automation for each type of model, and is also accountable for the graphic content generation. 
* And finally, the `figures` folder in which you may find all the schemes from the notebooks and other figures from the `README.md` file.

You may run the notebooks on Google Colaboratory in the following order :
1. `(1)_Data_formatting`
2. `(2)_Method_1`
3. `(3)_Method_2`
4. `(4)_Method_3`
5. `(5)_variables_strenght_assessment`
6. `(6)_zooplankton`

## Data

### Data recording
The measures were all collected in the North Sea area *in situ*, roughly between 1960 and 2010 at the rate of once a month, giving a total number of timesteps of 588. The recorded parameters are:
* Plankton datasets, acquired from the Coninuous Plankton Recorder (CPR) survey, and more particularly:
     * The **chlorophyll** (derived from the Phytoplankton Color Index, PCI).
     * The zooplankton, namely **total copepods** and **total *Calanus***.
* Physical parameters :
     * The **Sea Surface Temperature** (SST)
     * The **Mixed Layer Depth** (MLD)
     * The **heat losses**
     * The **wind**
* Climate indices : 
     * The **Nort Atlantic Oscillations** (NAO) index
     * The **Atlantic Multi-decadal Oscillations** (AMO) index

### Data formatting 
For consistency and with a view to performing future shifts predictions, we standardized the data using only P1 characteristics (mean and standard deviation).

## Methods comparison
### Method 1:
A Recurrent Neural Network (RNN) composed of a 3 hidden-layers Multi-Layers Perceptron (MLP), taking x(t) = [ Chl ; physics ] (t) as an input, and trained to output x(t+1). 

**NB :** During reconstructions, only the chlorophyll prediction is used and the neural network keeps being fed with the true physics. 

![shift](/figures/scheme_MLP_method1.PNG)

### Method 2:
The above RNN is not only trained on the prediction of x(t+1) but {x(t+1), x(t+2), ... , x(t+n)} with n ranging from 1 to 6. Thus, the network is iteratively applied on its own predictions, just like during the online reconstruction process. This method supposedly allows a better temporal patterns (like seasonal variablity) understanding. 

![shift](/figures/scheme_MLP_method2.PNG)

### Method 3:
In this method, the n previous consecutive states are "seen" by the MLP to allow the n forward states prediction. The input data and the target data are concatenated as [ x(t+1), x(t+2), ... , x(t+n) ].

![shift](/figures/scheme_MLP_method3.PNG)

### Leaderboard
The methods performances are not all equivalent. You may find below a brief summary:

![shift](/figures/leaderboard.PNG)

**NB:** 
* dt stands either for the number of previous and forward months (method 3) or the horizon forecast during training (method 2). 
* The MSE values are the average values of several models obtained with each of the methods. 
* One can observe that with a too high number of states, the performances of the 3rd method decrease. One idea could consist in assigning bigger weights in the loss to closer predictions (*i.e.* 1 month) and smaller to the furthest (*i.e.* 6 months). 


### Variables strenght assessment
