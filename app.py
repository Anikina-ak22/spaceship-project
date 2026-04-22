import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("train.csv")

st.title("🚀 Spaceship Titanic Prediction")

st.write("Прогноз чи буде пасажир Transported")

# Data cleaning
df = df.drop(["PassengerId", "Name", "Cabin"], axis=1)

X = df.drop("Transported", axis=1)
y = df["Transported"]

# categorical / numeric
cat_cols = X.select_dtypes(include="object").columns.tolist()
bool_cols = X.select_dtypes(include="bool").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

cat_cols += bool_cols

# Pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

st.subheader("📊 Accuracy")
st.write(round(acc, 3))

# Show dataset
if st.checkbox("Показати датасет"):
    st.dataframe(df.head(20))

# Prediction form
st.subheader("Введіть дані пасажира")

home = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
cryo = st.selectbox("CryoSleep", [True, False])
dest = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
age = st.slider("Age", 1, 80, 25)
vip = st.selectbox("VIP", [True, False])

room = st.number_input("RoomService", 0.0, 10000.0, 0.0)
food = st.number_input("FoodCourt", 0.0, 10000.0, 0.0)
shop = st.number_input("ShoppingMall", 0.0, 10000.0, 0.0)
spa = st.number_input("Spa", 0.0, 10000.0, 0.0)
vr = st.number_input("VRDeck", 0.0, 10000.0, 0.0)

if st.button("Передбачити"):

    sample = pd.DataFrame([{
        "HomePlanet": home,
        "CryoSleep": cryo,
        "Destination": dest,
        "Age": age,
        "VIP": vip,
        "RoomService": room,
        "FoodCourt": food,
        "ShoppingMall": shop,
        "Spa": spa,
        "VRDeck": vr
    }])

    result = model.predict(sample)[0]

    if result:
        st.success("Пасажир буде Transported")
    else:
        st.error("Пасажир НЕ буде Transported")