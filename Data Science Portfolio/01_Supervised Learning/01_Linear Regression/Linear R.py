import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load Superstore data (download from Tableau Public or use sample)
# Assuming you have 'superstore.csv'
df = pd.read_csv('C:/Users/USER/Desktop/Airflows/superstore.csv')

# Basic preprocessing
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
df = df.dropna(subset=['Profit', 'Sales', 'Discount'])

print(f"Dataset shape: {df.shape}")
print(df[['Sales', 'Profit', 'Discount']].describe())