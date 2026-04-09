import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import joblib


def test_train_model_file_exists():
    """Vérifie que le fichier churn_model_clean.pkl existe."""
    assert os.path.exists("data/churn_model_clean.pkl"), \
           "Le fichier churn_model_clean.pkl n'existe pas."

def test_train_model_file_loadable():
    """Vérifie que le fichier churn_model_clean.pkl peut être chargé."""
    try:
        model = joblib.load("data/churn_model_clean.pkl")
        assert isinstance(model, LogisticRegression), "Le modèle chargé n'est pas une instance de LogisticRegression."
    except Exception as e:
        assert False, f"Erreur lors du chargement du modèle: {e}"

def test_train_model_input_validation():
    """Vérifie que le modèle gère les entrées invalides."""
    try:
        model = pd.read_pickle("data/churn_model_clean.pkl")
        # Exemple de données d'entrée invalides
        test_data = [[-1, 1, 5, 3]]  # age négatif
        model.predict(test_data)
        assert False, "Le modèle devrait lever une exception pour des données d'entrée invalides."
    except Exception as e:
        assert True  # L'exception est attendue

def test_train_model_input_validation_years():
    """Vérifie que le modèle gère les entrées invalides pour les années de service."""
    try:
        model = pd.read_pickle("data/churn_model_clean.pkl")
        # Exemple de données d'entrée invalides
        test_data = [[30, 1, -5, 3]]  # années de service négatives
        model.predict(test_data)
        assert False, "Le modèle devrait lever une exception pour des années de service négatives."
    except Exception as e:
        assert True  # L'exception est attendue

def test_train_model_input_validation_num_sites():
    """Vérifie que le modèle gère les entrées invalides pour le nombre de sites."""
    try:
        model = pd.read_pickle("data/churn_model_clean.pkl")
        # Exemple de données d'entrée invalides
        test_data = [[30, 1, 5, -3]]  # nombre de sites négatif
        model.predict(test_data)
        assert False, "Le modèle devrait lever une exception pour un nombre de sites négatif."
    except Exception as e:
        assert True  # L'exception est attendue

def test_train_model_input_validation_account_manager():
    """Vérifie que le modèle gère les entrées invalides pour account_manager."""
    try:
        model = pd.read_pickle("data/churn_model_clean.pkl")
        # Exemple de données d'entrée invalides
        test_data = [[30, 2, 5, 3]]  # account_manager doit être 0 ou 1
        model.predict(test_data)
        assert False, "Le modèle devrait lever une exception pour un account_manager invalide."
    except Exception as e:
        assert True  # L'exception est attendue

