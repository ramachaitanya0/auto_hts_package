# Auto Hierarchical Time Series Regressor
---
This package is an extension of the [scikit-hts](https://scikit-hts.readthedocs.io/en/latest/readme.html) package . 
we have developed this package in order to provide extra functionalities for the scikit-hts [Regressor](https://scikit-hts.readthedocs.io/en/latest/hts.html#hts.HTSRegressor).
In scikit-hts Regressor class, we can use one Forecasting model at a time for all the nodes of hierarchy. Same forecasting model may not fit well for all the nodes in the hierarchy. Using a single model for all the nodes, we are unable to capture and take advantage of individual series characteristics such as time dynamics, special events, different seasonal patterns, etc.
In this package, we are providing a solution for it by applying all the available forecasting models on all the nodes of hierarchy and find the best model for each node of hierarchy. After selecting the models and predicting the forecasts for each node, we run reconciliation method on the forecasted data to make the data coherent.


## Installation
--- 
Few modules used in this package are not supported by python >= 3.9.* . So, we recommend you to create an virtual environment with python=3.8.* and run requirements.txt file 

Step 1 :

Create virtual environment using conda and activate the environment
```bash
conda create -n <env_name> python=3.8
```
```bash
conda activate <env_name>
```

Step 2 :
```bash
pip install -r requirements.txt
```
Step 3 :
```bash
pip install auto_hts_forecast
```
## Usage
--- 
Refer to [demo.ipynb](https://github.com/ramachaitanya0/auto_hts_package/blob/main/test/demo.ipynb) in test directory  




