# FAANG-stock-predicition
stock prediction of FAANG using python libraries and ML transformations
Code written by Agnes Folga 
1. Dataset
FAANG (FB,Amazon,Apple,Netflix,Google) Stocks 📈 | Kaggle
This dataset from Kaggle features the open-high-low-close (OHLC) data of the top 5
technology companies stock in the US. Each company has its own CSV file with
columns Data, Opening price of the day, Highest price of the day, Lowest price of the
day, Closing price of the day, and Trading volume for the day.
2. Task/Problem
Using this data, we can try to predict the closing stock price for each company. 
Possible preprocessing techniques include firstly eliminating any outlier data points.
Then, we will trim the dataset by only keeping the closing data (float) and date (Datetime
Index) columns. Lastly, we must conduct a train-test split (80/20 split). A good solution is
to also use cross validation. “The idea is simple: we split the training data into K folds;
then, for each fold k ∈ {1,...,K}, we train on all the folds but the k’th, and test on the k’th,
in a round-robin fashion… We then compute the error averaged over all the folds, and
use this as a proxy for the test error.“ 
We can attempt this in python, and can complete this on my computer (personal
machine). Python libraries include pandas dataframe, scikit-learn, sk-learn, numpy. The
input in this problem are the given CSV files for each company, and the desired output is
an accurate prediction of the closing stock price for each company.
This problem is very pertinent to the population and computer science students in
general as this model can possibly predict the stock price for tech companies they have
an interest in working for. Many individuals have an interest in the stock market, and this
could be useful to them. Beyond individuals with an interest, actual investment bankers
and potential investors can try to maximize their financial payoffs by using such models.
3. Plausible ML technique
We can use a linear regression model to predict the closing stock price for each
company. In machine learning, we must use 2 sets of data to have the model be
effective: testing and training data. The scikit-learn test_train_split function in python will
be able to split our data into test (20%) and training (80%) sets. After this is done, we
can train the model using the Linear Regression Model from sklearn.linear_model in
python. Using the model we can predict the closing stock prices.
We will use the linear regression models to calculate the mean squared error (the
average of squared error between the predicted values and actual values), the model
coefficients, and the coefficient of determination. A lower MSE would be ideal, and the
correlation coefficient should be close to 1 if it is a good model.
Possible modifications or different techniques include a Lasso/Ridge regression, or a
Support Vector Algorithm (SVM)
