import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine

#Konfiguracja połączenia z serwerem MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "a7d3916714c94a19ada9cf91e3627b56"
model_uri = f"runs:/{RUN_ID}/model"

try:
    #Ładowanie modelu
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print("Model został pomyślnie załadowany")
except Exception as e:
    print("Błąd: Nie można załadować modelu")
    exit()

#Przygotowanie przykładowej próbki ze zbioru Wine
wine = load_wine()

sample_raw = wine.data[0:1]
sample_df = pd.DataFrame(sample_raw, columns=wine.feature_names)

#Wykonanie predykcji
prediction = loaded_model.predict(sample_df)
probabilities = loaded_model.predict_proba(sample_df)

print("Wyniki predykcji:")
print(f"Klasa przewidziana: {prediction[0]} (Nazwa: {wine.target_names[prediction[0]]})")
print(f"Prawdziwa klasa:    {wine.target[0]}")