# Batting Average Estimator: Project Overview
* Created a tool that estimates your batting average (MAE ~ 0.008) based on other features relative to batting statistics in baseball (variance score of 88%)
* Optimized Linear, and Random Forest Regressors using GridsearchCV to reach the best model to use for this project
* Looked at features from 2019 MLB Season (batters) to use valid predictions
* Built a client facing API using Flask

# Code and Resources Used
Python Version: 3.7
Packages: pandas, numpy, sklearn, matplotlib, seaborn, flask, pickle

Data set: https://www.baseball-reference.com/leagues/MLB/2019-standard-batting.shtml

# Data Collection and Cleaning
* collected data from mlb 2019 seasons batters only
* removed rows of players traded during season to avoid confusion
* changed data to provide only batters with more than 300 plate appearance for better accuracy
* added columns to provide age over 30 to see if age affects your ability to hit
* Cleaned null data and removed data that is irrelevant to us

# EDA
I looked at various distribution plots of the data to account for some of the theory behind how a players batting average is determined. Below are a few plots.

![alt text](https://github.com/Wellington77/mlb_batters_proj/blob/master/AgetoBA.png)
![alt text](https://github.com/Wellington77/mlb_batters_proj/blob/master/At_batstoBA.png)
![alt text](https://github.com/Wellington77/mlb_batters_proj/blob/master/Heatmap.png)

# Model Building 
I split the categorical variables in the training and testing sets with a test size of 20%

I tried two different models and evaluated them using Mean Absolute Error. (I chose MAE because it's easier to intepret and useful for this case)

Models I tried were: Linear Regression and Random Forest Regression.

# Model Performance
The Linear Regression model outperformed the Random Forest Regressor. This will bring a better evaluation when predicting batting average

* Linear Regression MAE = 0.008
* Random Forest Regression MAE = 0.01
* corss validation score of 88%

# Productionizing
In this step I build a Flask API that was hosted on a local webserver. This API endpoint takes in various request and returns your batting average. 

