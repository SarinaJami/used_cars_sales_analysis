# used_cars_sales_analysis
The dataset includes information about used cars and their prices. 

The task is to train a model to predict the prices of the used cars based on the information provided such as odometer, fuel, age, type, etc.

First, we want to understand the data by visualization and statistics (Data Analysis and Manipulation). After fully comprehending the data and its shortcomings, we start the preprocessing phase which includes changing data types, filling missing values, drop cols and rows, create new features, and make sure the variables are independent and ready for training a model.

Then, we start by fitting a linear regression to grasp an understanding of the relation between the dependent and the independent variables (input and target). After that, we fit a non-linear regression model by adding polynomial features to out dataset (feature engineering), which yields lower error but still cannot fit the data well. Finally, we train a Random Forest Regressor and after applying cross-validation on our data, we use grid search to find the best parameters for our model. The best estimator provides a model with an RMSE almost as half of RMSE of the non-linear regressor model.

The whole Python code is available in the car_sales_analysis.py file. I used Spyder for my project.
