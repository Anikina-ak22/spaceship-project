import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import plotly.express as px
import plotly.figure_factory as ff

# PAGE SETTINGS
st.set_page_config(
    page_title="Spaceship Titanic ML App",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Spaceship Titanic Prediction App")
st.write("Інтерактивний ML веб-додаток для прогнозування Transported")

# LOAD DATA
df = pd.read_csv("train.csv")

# SAVE ORIGINAL FOR VISUALIZATION
df_visual = df.copy()

# CLEAN DATA FOR MODEL
df = df.drop(["PassengerId", "Name", "Cabin"], axis=1)

X = df.drop("Transported", axis=1)
y = df["Transported"]

# COLUMN TYPES
cat_cols = X.select_dtypes(include="object").columns.tolist()
bool_cols = X.select_dtypes(include="bool").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

cat_cols += bool_cols

# PIPELINE
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
        n_estimators=250,
        random_state=42
    ))
])

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

# SIDEBAR
st.sidebar.title("Навігація")

section = st.sidebar.radio(
    "Оберіть розділ:",
    [
        "Головна",
        "Візуалізація даних",
        "Метрики моделі",
        "Передбачення"
    ]
)

# HOME
if section == "Головна":

    st.subheader("Про проект")

    st.write("""
    Цей додаток прогнозує, чи буде пасажир переміщений
    в інший вимір (Transported) на космічному кораблі Titanic.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Кількість рядків", df.shape[0])
    col2.metric("Кількість ознак", df.shape[1])
    col3.metric("Accuracy", round(acc, 3))

    st.dataframe(df_visual.head(20))


# VISUALIZATION
elif section == "Візуалізація даних":

    st.subheader("📊 Візуалізація датасету")

    # TARGET DISTRIBUTION
    fig1 = px.histogram(
        df_visual,
        x="Transported",
        color="Transported",
        title="Розподіл цільової змінної"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # AGE
    fig2 = px.histogram(
        df_visual,
        x="Age",
        nbins=30,
        title="Розподіл віку пасажирів"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # HOME PLANET
    fig3 = px.pie(
        df_visual,
        names="HomePlanet",
        title="Пасажири за планетами"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # DESTINATION
    fig4 = px.bar(
        df_visual["Destination"].value_counts().reset_index(),
        x="Destination",
        y="count",
        title="Популярність напрямків"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # SPENDING
    spend_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    spend_mean = df_visual[spend_cols].mean().reset_index()
    spend_mean.columns = ["Service", "Mean"]

    fig5 = px.bar(
        spend_mean,
        x="Service",
        y="Mean",
        title="Середні витрати по сервісах"
    )
    st.plotly_chart(fig5, use_container_width=True)


# METRICS
elif section == "Метрики моделі":

    st.subheader("📈 Якість моделі")

    st.metric("Accuracy", round(acc, 3))

    # Confusion Matrix
    z = cm

    x = ["False", "True"]
    y_labels = ["False", "True"]

    fig = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y_labels,
        colorscale="Blues"
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Classification report
    report = classification_report(y_test, pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.write("Classification Report")
    st.dataframe(report_df)


# PREDICTION
elif section == "Передбачення":

    st.subheader("🔮 Введіть параметри пасажира")

    col1, col2 = st.columns(2)

    with col1:
        home = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
        cryo = st.selectbox("CryoSleep", [True, False])
        dest = st.selectbox(
            "Destination",
            ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
        )
        age = st.slider("Age", 1, 80, 25)
        vip = st.selectbox("VIP", [True, False])

    with col2:
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
        proba = model.predict_proba(sample)[0]

        if result:
            st.success("Пасажир буде Transported")
        else:
            st.error("Пасажир НЕ буде Transported")

        st.write("Ймовірність:")
        st.write(f"False: {round(proba[0]*100,2)}%")
        st.write(f"True: {round(proba[1]*100,2)}%")