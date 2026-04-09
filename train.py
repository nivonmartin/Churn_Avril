import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib


# Récupération des données
filename = "customer_churn.csv"
data_dir = "data"
filepath = os.path.join(data_dir, filename)
data = pd.read_csv(filepath)

# Séparer X et y
X = data[["Age", "Account_Manager", "Years", "Num_Sites"]]
y = data["Churn"]

# Création et entrainement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Sauvegarder le modèle entraîné avec joblib
model_filename = "churn_model_clean.pkl"
model_filepath = os.path.join(data_dir, model_filename)
joblib.dump(model, model_filepath)

print("Modèle de régression logistique entrainé et sauvegardé !")

# # Splitter les données en train et test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model.fit(X_train, y_train)

# # Prédiction et score du modèle
# y_pred = model.predict(X_test)

# print(f"Accuracy  : {accuracy_score(y_test, y_pred)}")
# print(f"Recall    : {recall_score(y_test, y_pred)}")
# print(f"Precision : {precision_score(y_test, y_pred)}")
# print(f"F1-score  : {f1_score(y_test, y_pred)}")
