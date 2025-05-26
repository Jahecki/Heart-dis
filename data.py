import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib



df = pd.read_csv('heart_cleveland_upload.csv')

print("Dane wczytane!")
print(df.head())


X = df.drop("condition", axis=1)
y = df["condition"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


score = clf.score(X_test, y_test)
print(f"Dokładność na zbiorze testowym: {score:.2f}")


joblib.dump(clf, "model.pkl")
print("Model zapisany w pliku: model.pkl ✅")