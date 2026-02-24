import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Start MLflow experiment
with mlflow.start_run():
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    #Log parameters & metrics
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc) # type: ignore
    
    #Log model
    mlflow.sklearn.log_model(model, "model") # type: ignore
    
    print("Accuracy:", acc)
    
    joblib.dump(model, "model.pkl")
    print("Model saved successfully")