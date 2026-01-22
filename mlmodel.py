import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Charger les données
df = pd.read_csv('FuelConsumption.csv')

# Sélectionner les attributs requis
features = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
X = df[features]
y = df['CO2EMISSIONS']

# Créer un modèle de régression polynomiale de degré 3
model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_regression', LinearRegression())
])

# Entraîner le modèle
model.fit(X, y)

# Sauvegarder le modèle avec pickle
with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)

print("Modèle de régression polynomiale (degré 3) entraîné et sauvegardé dans 'model.pickle'")