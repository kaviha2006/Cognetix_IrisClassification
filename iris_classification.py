import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------
# LOAD THE DATASET FROM iris.data
# -----------------------------------
df = pd.read_csv("iris.data", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# -----------------------------------
# ENCODE TEXT LABELS (Setosa, Versicolor, Virginica)
# -----------------------------------
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# FEATURES AND TARGET
X = df.iloc[:, :4]
y = df["species"]

# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# MODEL TRAINING (Decision Tree)
# -----------------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -----------------------------------
# TERMINAL UI
# -----------------------------------
print("\n====== IRIS FLOWER CLASSIFICATION ======")
print("Enter flower measurements to predict the species.\n")

sl = float(input("Enter Sepal Length (cm): "))
sw = float(input("Enter Sepal Width  (cm): "))
pl = float(input("Enter Petal Length (cm): "))
pw = float(input("Enter Petal Width  (cm): "))

# -----------------------------------
# PREDICT
# -----------------------------------
prediction = model.predict([[sl, sw, pl, pw]])[0]
species_name = le.inverse_transform([prediction])[0]

print("\n----------------------------------------")
print(" Predicted Species:", species_name)
print("----------------------------------------")

accuracy = model.score(X_test, y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
print("----------------------------------------\n")
