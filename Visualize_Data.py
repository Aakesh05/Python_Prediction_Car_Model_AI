import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# Using Panda to import the given dataset "Net_Worth_Data.xlsx"
data = pd.read_excel(r'C:\Users\schoo\Documents\TECHTORIUM !!!\2023\Term 3\TERM 3 ASSESSMENTS\Net_Worth_Data.xlsx')

# This code determines the datasets shape
num_rows, num_columns = data.shape

# This code shows the summary of the dataset
data_info = data.info()

# We are hecking the null values in dataset so we can handle this later if any
null_values = data.isnull().sum()

# Plotting the graph to show relations with the different coloums
# We then select the independent variables, target variables, and irrelevant features.
sns.pairplot(data)
plt.show()