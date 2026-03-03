
import pandas as pd
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt

conn=sqlite3.connect("./data/inventory.db")

tables=pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'",conn)

for table in tables['name']:
  print(f"table name : {table}")
  df=pd.read_sql_query(f"SELECT * FROM {table}",conn)
  display(df.head())

# after looking at each tables we are using the vendor_invoice table as our base table
vendor_df=pd.read_sql_query("SELECT * FROM vendor_invoice",conn)
vendor_df.head()

# now we need to define the correlation b/w the attributes to find the relation shared betweeen them
vendor_df[["Quantity","Freight","Dollars"]].corr()

#we find that quantitty and freight have a pretty close relations
# for better visualization we will use the heat map and the sccatter pllot

plt.figure(figsize=(4,2))
sns.heatmap(vendor_df[["Quantity","Freight","Dollars"]].corr(),annot=True)

plt.figure(figsize=(4,2))
plt.scatter(vendor_df["Quantity"],vendor_df['Freight'])
plt.scatter(vendor_df["Dollars"],vendor_df['Freight'])
plt.legend(["Quantity","Dollars"])
plt.xlabel("Quantity/Dollars")
plt.ylabel("Freight")
plt.plot()
plt.show()

# we need to calculate the freight per unit
vendor_df['freight_per_unit']=vendor_df['Freight']/vendor_df['Quantity']
vendor_df.head()

#finding the data for the lower and the upper values of the order
#in order to locate the lower and upper bucket of the order
low_quantity=vendor_df['Quantity'].quantile(0.25)
high_quantity=vendor_df['Quantity'].quantile(0.75)
low_quantity
high_quantity

#filtering the data for the pure findings of the needed data
vendor_df.loc[vendor_df['Quantity']<low_quantity,'freight_per_unit'].mean()

vendor_df.loc[vendor_df['Quantity']>high_quantity,'freight_per_unit'].mean()

#spliiting the dataset
X=vendor_df[['Dollars']]  #the double braces is required since the linear refression model requires 2d parameters
y=vendor_df['Freight']

vendor_df.describe().round()

#from the above operations we find that there is a huge diffrence of 0.05 between the low quantity and high quantity
#the data states that there are huge possible of the outlieers being present
#but we wont be removing the outliers since there might exist some users who only orders in bulks

from sklearn.model_selection import train_test_split

#we will be having a 80-20 data split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#we can try out our diffrent models to find out the best model we can appply for the process
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

#testing out our diffrent models
model1=LinearRegression()
model1.fit(X_train,y_train)


model2=DecisionTreeRegressor(random_state=42)
model2.fit(X_train,y_train)

model3=RandomForestRegressor(random_state=42)
model3.fit(X_train,y_train)

import numpy as np
def evaluate_model(model,X_test,y_test,model_name):
  preds=model.predict(X_test)
  mae=mean_absolute_error(y_test,preds)
  rmse=np.sqrt(mean_squared_error(y_test,preds))
  r2=r2_score(y_test,preds)*100
  print(f"{model_name} Perforamnce")
  print(f"mae : {mae:.2f}")
  print(f"rmse : {rmse:.2f}")
  print(f"r2 : {r2:.2f}")

evaluate_model(model1,X_test,y_test,"linear regression")
evaluate_model(model2,X_test,y_test,"decision tree regression")
evaluate_model(model3,X_test,y_test,"random forest regression")

#We found that the model1 performs good
#simple model performs better than the other 2 complex models
#so we need to adjust the max_depths , in order to try out the best of the best models
model1=LinearRegression()
model1.fit(X_train,y_train)


model2=DecisionTreeRegressor(max_depth=4,random_state=42)
model2.fit(X_train,y_train)

model3=RandomForestRegressor(max_depth=4,random_state=42)
model3.fit(X_train,y_train)

evaluate_model(model1,X_test,y_test,"linear regression")
evaluate_model(model2,X_test,y_test,"decision tree regression")
evaluate_model(model3,X_test,y_test,"random forest regression")

#visualising using the scatter plot
plt.scatter(X_test,y_test)
plt.plot(X_test,model1.predict(X_test),color='red')
plt.show()

# to use the model
input_data={
    "Dollars":[18500,9000]
}
df=pd.DataFrame(input_data)

model1.predict(df)

