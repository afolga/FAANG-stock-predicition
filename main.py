import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


netflix=pd.read_csv('C:\\Users\\agnes\\Documents\\CS412\\Project\\stock_csvs\\NFLX Historical Data.csv')
apple=pd.read_csv('C:\\Users\\agnes\\Documents\\CS412\\Project\\stock_csvs\\AAPL Historical Data.csv')
amazon=pd.read_csv('C:\\Users\\agnes\\Documents\\CS412\\Project\\stock_csvs\\AMZN Historical Data.csv')
google=pd.read_csv('C:\\Users\\agnes\\Documents\\CS412\\Project\\stock_csvs\\GOOGL Historical Data.csv')
facebook=pd.read_csv('C:\\Users\\agnes\\Documents\\CS412\\Project\\stock_csvs\\FB Historical Data.csv')

netflix=netflix[['Date','Close']]
apple=apple[['Date','Close']]
amazon=amazon[['Date','Close']]
google=google[['Date','Close']]
facebook=facebook[['Date','Close']]
#NETFLIX
X_netflix=netflix['Close']
Y_netflix=netflix['Date']
x_train, x_test, y_train, y_test=train_test_split(X_netflix, Y_netflix, test_size=0.2)
clf=LinearRegression()
x_train= x_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
for i in range(len(y_train)):
    y_train[i]=float(y_train[i][0].replace('-',""))
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
clf.fit(x_train, y_train)
clf.predict(x_test)
for i in range(len(y_test)):
    y_test[i]=float(y_test[i][0].replace('-',""))
netflix_accuracy= clf.score(x_test, y_test)
print('NETFLIX ACCURACY:' +str(netflix_accuracy))

#APPLE
x_apple=apple['Close']
y_apple=apple['Date']
x_train, x_test, y_train, y_test=train_test_split(x_apple, y_apple, test_size=0.2) #have to make dates into floats
clf=LinearRegression()
x_train= x_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
for i in range(len(y_train)):
    y_train[i]=float(y_train[i][0].replace('-',""))
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
clf.fit(x_train, y_train)
clf.predict(x_test)
for i in range(len(y_test)):
    y_test[i]=float(y_test[i][0].replace('-',""))
apple_accuracy= clf.score(x_test, y_test)
print('APPLE ACCURACY:' +str(apple_accuracy))


#AMAZON
x_amazon=amazon['Close']
y_amazon=amazon['Date']
x_train, x_test, y_train, y_test=train_test_split(x_amazon, y_amazon, test_size=0.2)
clf=LinearRegression()
x_train= x_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
for i in range(len(y_train)):
    y_train[i]=float(y_train[i][0].replace('-',""))
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
clf.fit(x_train, y_train)
clf.predict(x_test)
for i in range(len(y_test)):
    y_test[i]=float(y_test[i][0].replace('-',""))
amazon_accuracy= clf.score(x_test, y_test)
print('AMAZON ACCURACY:' +str(amazon_accuracy))


#GOOGLE
x_google=google['Close']
y_google=google['Date']
x_train, x_test, y_train, y_test=train_test_split(x_google, y_google, test_size=0.2)
clf=LinearRegression()
x_train= x_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
for i in range(len(y_train)):
    y_train[i]=float(y_train[i][0].replace('-',""))
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
clf.fit(x_train, y_train)
clf.predict(x_test)
for i in range(len(y_test)):
    y_test[i]=float(y_test[i][0].replace('-',""))
google_accuracy= clf.score(x_test, y_test)
print('GOOGLE ACCURACY:' +str(google_accuracy))


#FACEBOOK
x_facebook=facebook['Close']
y_facebook=facebook['Date']
x_train, x_test, y_train, y_test=train_test_split(x_facebook, y_facebook, test_size=0.2)
clf=LinearRegression()
x_train= x_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
for i in range(len(y_train)):
    y_train[i]=float(y_train[i][0].replace('-',""))
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
clf.fit(x_train, y_train)
clf.predict(x_test)
for i in range(len(y_test)):
    y_test[i]=float(y_test[i][0].replace('-',""))
facebook_accuracy= clf.score(x_test, y_test)
print('FACEBOOK ACCURACY:' +str(facebook_accuracy))
