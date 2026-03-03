from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


def train_linear_regression(X_train, y_train):
  model=LinearRegression()
  model.fit(X_train,y_train)
  return model


def train_decision_tree(X_train, y_train,max_depth=5):
  model=DecisionTreeRegressor(max_depth=max_depth,random_state=42)
  model.fit(X_train,y_train)
  return model

def train_random_forest(X_train, y_train,max_depth=5):
  model=RandomForestRegressor(
      max_depth=max_depth,random_state=42
  )
  model.fit(X_train,y_train)
  return model


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
  return mae,rmse,r2



