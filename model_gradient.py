import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform

# 1. Charger les données
df = pd.read_csv("carData.csv")

# 2. Feature engineering
df["Car_Age"] = 2025 - df["Year"]
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# 3. Définir X et y
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# 4. Préparation des transformations
numerical_features = ["Present_Price", "Kms_Driven", "Car_Age"]
categorical_features = ["Fuel_Type", "Seller_Type", "Transmission", "Owner"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])

# 5. Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Pipeline de base
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

# 7. Définir les hyperparamètres à tester
param_grid = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [3, 4, 5, 6],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.6, 0.8, 1.0]
}

param_dist = {
    "regressor__n_estimators": randint(100, 500),
    "regressor__max_depth": randint(3, 10),
    "regressor__learning_rate": uniform(0.01, 0.3),
    "regressor__subsample": uniform(0.5, 0.5)
}

# 8. Recherche aléatoire (Random Search)
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
print("✅ Meilleurs paramètres (Random Search) :", random_search.best_params_)

# 9. Recherche exhaustive (Grid Search)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("✅ Meilleurs paramètres (Grid Search) :", grid_search.best_params_)

# 10. Utiliser le meilleur modèle trouvé (choix : Random ou Grid)
best_model = random_search.best_estimator_  # ou grid_search.best_estimator_

# 11. Évaluation
y_pred = best_model.predict(X_test)
print("📊 Gradient Boosting Optimisé")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 12. Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Gradient Boosting Optimisé - Réel vs Prédit")
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. Sauvegarde du meilleur modèle
joblib.dump(best_model, "modele_voiture.pkl")
print("✅ Modèle exporté dans 'modele_voiture.pkl'")
