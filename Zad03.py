import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
#Załadowanie zbioru danych Wine
data = load_wine()
X = data.data
y = data.target

#Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Definicja hiperparametrów
max_depth = 3

#Ustawienie nazwy eksperymentu
mlflow.set_experiment("WineRandomForest")

for i in [1, 2, 3, 7, 15, 30]:
    max_depth = i
    #Rozpoczęcie sesji MLflow
    with mlflow.start_run():
        #Inicjalizacja modelu (RandomForestClassifier zamiast DecisionTree)
        rf = RandomForestClassifier(
            max_depth = max_depth,
            random_state = 42
        )

        #Trenowanie modelu
        rf.fit(X_train, y_train)

        #Predykcje
        y_pred = rf.predict(X_test)

        #Obliczanie metryki accuracy
        acc = accuracy_score(y_test, y_pred)

        #Zapisywanie hiperparametrów
        mlflow.log_param("max_depth", max_depth)
        # mlflow.log_param("criterion", criterion)

        #Logowanie metryki accuracy
        mlflow.log_metric("accuracy", acc)

        #Logowanie modelu wraz z artefaktami
        mlflow.sklearn.log_model(rf, name="model")

    print(f"Trening zakończony!\n"
          f"Dokładność: {acc:.4f}")

