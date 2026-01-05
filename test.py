import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



from RandomForestClassifier import RandomForestCustom

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Benchmark on {len(X_train)} samples...")
print("-" * 30)


start_custom = time.perf_counter() 

my_rf = RandomForestCustom(n_estimators=100, random_state=42)
my_rf.fit(X_train, y_train)
my_preds = my_rf.predict(X_test)

end_custom = time.perf_counter()
durée_custom = end_custom - start_custom

print(f"RandomForestCUstom :")
print(f"Precision : {accuracy_score(y_test, my_preds):.4f}")
print(f"Time : {durée_custom:.4f} s")
print("-" * 30)


start_sk = time.perf_counter()

sk_rf = RandomForestClassifier(n_estimators=100, random_state=42)
sk_rf.fit(X_train, y_train)
sk_preds = sk_rf.predict(X_test)

end_sk = time.perf_counter()
durée_sk = end_sk - start_sk

print(f"SKLEARN RandomForest :")
print(f"Precision : {accuracy_score(y_test, sk_preds):.4f}")
print(f"Time : {durée_sk:.4f} s")
