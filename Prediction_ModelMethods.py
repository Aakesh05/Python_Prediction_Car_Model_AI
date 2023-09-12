import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np


def Import_Data(Net_Worth_Data):
    return pd.read_excel(r"C:\Users\schoo\Documents\TECHTORIUM !!!\2023\Term 3\TERM 3 ASSESSMENTS\Net_Worth_Data.xlsx")


def Preprocessing_Dataset(data):
    data = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Gender', 'Age', 'Income', 'Credit Card Debt', 'Healthcare Cost', 'REITs'], axis=1)
    
    X = data.drop(columns=['Net Worth'])
    Y = data['Net Worth']
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1


def Splitting_Dataset(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


def Training_Regression_Models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train.ravel())
        models[name] = model
        
    return models

def Evaluating_Performance(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    return rmse_values



def Plotting_Performance(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'black'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    



def Saving_Best_Performance_Model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    dump(best_model, "net_worth_model.joblib")
    
def Loaded_Model(loaded_model):
    loaded_model = net_worth_model.joblib

def Gather_User_Inputs():

    Inherited = int(input("Enter inherited amount: "))

    Stocks = int(input("Enter stock value: "))

    Bonds = float(input("Enter bonds amount: "))

    Mutual_Funds = float(input("Enter mutual funds: "))

    ETFs = float(input("Enter ETFs value: "))

   

    return Inherited, Stocks, Bonds, Mutual_Funds, ETFs

 

def Scale_User_Inputs(user_inputs_scaled, sc):

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        user_input_scaled = sc.transform([user_inputs])


    return user_input_scaled
   
  
 
  
    
    


if __name__ == "__main__":
    try:
        # Load data
        data = Import_Data(r"C:\Users\schoo\Documents\TECHTORIUM !!!\2023\Term 3\TERM 3 ASSESSMENTS\Net_Worth_Data.xlsx")
    
        #Preproccess Data
        X_scaled, y_scaled, sc, sc1 = Preprocessing_Dataset(data)
        X_train, X_test, y_train, y_test = Splitting_Dataset(X_scaled, y_scaled)
        
        # Train models and evaluate
        models = Training_Regression_Models (X_train, y_train)
        rmse_values = Evaluating_Performance(models, X_test, y_test)
        Plotting_Performance(rmse_values)
        Saving_Best_Performance_Model(models, rmse_values)
        
        # Load the best model
        Loaded_Model = Loaded_Model
        
        # Gather user inputs
        Gather_User_Inputs = Gather_User_Inputs()
        
        # Scale user inputs
        Scale_User_Inputs = Scale_User_Inputs(user_input_scaled, sc)
        
        # Make prediction using the loaded model
        predicted_amount = predict_with_model(loaded_model, scaled_inputs, sc1)
        print("Predicted Net Worth:", predicted_amount[0])
        
    except ValueError as ve:
        print("Error: {ve}")