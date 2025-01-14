import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load Titanic data."""
    return pd.read_csv(path)

def preprocess(data):
    """Preprocess the Titanic data."""
    data = data.fillna(0)
    data = pd.get_dummies(data, drop_first=True)
    return data

def train_model(train_path):
    """Train a RandomForest model on Titanic data."""
    data = load_data(train_path)
    data = preprocess(data)
    
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    
    return model

if __name__ == "__main__":
    train_model("data/raw/train.csv")
