import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load


import numpy as np

def Main():
    # Import the dataset
    data = pd.read_excel(r"C:\Users\schoo\Documents\TECHTORIUM !!!\2023\Term 3\TERM 3 ASSESSMENTS\Net_Worth_Data.xlsx")


    # Drop the Irrelavant Columns
    data = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Gender', 'Age', 'Income', 'Credit Card Debt', ' Healthcare Cost', 'REITS'], axis=1)

    # Split data into their designated set                                    
    X = data.drop(colomns=['Net Worth'])
    Y = data['Net Worth']
    
    # Data Preprocessing
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Train a Model
    Regression_Model = LinearRegression()
    model.fit(X_Scaled, Y)

    # Saving the model
    joblib.dump(Regression_Model, 'net_worth_model.pkl')
    
    # Gather user inputs
    InheritedAmount = int(input("Enter inherited Amounts: "))
    Stocks = int(input("Enter stock value: "))
    Bonds = float(input("Enter bonds amount: "))
    MutualFunds = float(input("Enter Mutual funds: "))
    ETFs = float(input("Enter ETFs Value: "))
    
    # Scale User Inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        user_input_scaled = scaler_X.transform([[InheritedAmount, Stocks, Bonds, MutualFunds, ETFs]])
        
    # Now make a prediction
    predicted_amount - loaded_model.predict(user_input_scaled)
    print("Predicted Net Worth Is:", predicted_amount[0])
    

if __name__ == "__main__":
    try:
        # Load data
        data = pd.read_excel(r"C:\Users\schoo\Documents\TECHTORIUM !!!\2023\Term 3\TERM 3 ASSESSMENTS\Net_Worth_Data.xlsx")
    
        #Preproccess Data
        X_scaled, y_scaled, sc, sc1 = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
        
        # Train models and evaluate
        models = train_models (X_train, y_train)
        rmse_values
        evaluate_models (models, X_test, y_test)
        plot_model_performance(rmse_values)
        save_best_model(models, rmse_values)
        
        # Load the best model
        loaded_model = load_best_model()
        
        # Gather user inputs
        user_inputs = gather_user_inputs()
        
        # Scale user inputs
        scaled_inputs = scale_user_inputs(user_inputs, sc)
        
        # Make prediction using the loaded model
        predicted_amount = predict_with_model(loaded_model, scaled_inputs, sc1)
        print("Predicted Net Worth:", predicted_amount[0])
        
    except ValueError as ve:
        print("Error: {ve}")