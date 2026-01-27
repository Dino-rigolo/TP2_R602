import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv('FuelConsumption.csv')

# Sélectionner les attributs requis
features = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
X = df[features]
y = df['CO2EMISSIONS']

# Diviser en données d'entraînement et test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False, random_state=0)

# Créer un modèle de régression polynomiale de degré 3 avec normalisation
polynomial_reg = Pipeline([
    ('std_scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_regression', LinearRegression())
])

# Entraîner le modèle
polynomial_reg.fit(x_train, y_train)

# Afficher les performances
train_score = polynomial_reg.score(x_train, y_train)
test_score = polynomial_reg.score(x_test, y_test)

print(f"Score d'entraînement: {train_score:.4f}")
print(f"Score de test: {test_score:.4f}")

# Sauvegarder le modèle avec pickle
with open('model.pickle', 'wb') as file:
    pickle.dump(polynomial_reg, file)

print("Modèle de régression polynomiale (degré 3) entraîné et sauvegardé dans 'model.pickle'")

# Exemple de prédiction
data = pd.DataFrame([[2014, 2, 4, 5]], columns=features)
prediction = polynomial_reg.predict(data)
print(f"Prédiction exemple: {prediction[0]:.2f}")