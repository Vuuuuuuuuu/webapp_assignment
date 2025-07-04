import os
import pickle
import streamlit as st
import pandas as pd
from os import path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("beer-servings(1).csv")
print(df.head())
print(df.shape)
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDescriptive statistics for numerical features:")
print(df.describe())


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df["beer_servings"], kde=True)
plt.title("Distribution of Beer Servings")

plt.subplot(2, 2, 2)
sns.histplot(df["spirit_servings"], kde=True)
plt.title("Distribution of Spirit Servings")

plt.subplot(2, 2, 3)
sns.histplot(df["wine_servings"], kde=True)
plt.title("Distribution of Wine Servings")

plt.subplot(2, 2, 4)
sns.histplot(df["total_litres_of_pure_alcohol"], kde=True)
plt.title("Distribution of Total Liters of Pure Alcohol")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df["continent"].value_counts().plot(kind="bar")
plt.title("Continent Distribution")

plt.subplot(1, 2, 2)
df["country"].value_counts().head(10).plot(kind="bar")  # Show top 10 countries
plt.title("Top 10 Countries by Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(
    df[
        [
            "beer_servings",
            "spirit_servings",
            "wine_servings",
            "total_litres_of_pure_alcohol",
        ]
    ].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

plt.figure(figsize=(15, 6))
numerical_cols = [
    "beer_servings",
    "spirit_servings",
    "wine_servings",
    "total_litres_of_pure_alcohol",
]
for i, col in enumerate(numerical_cols):
    plt.subplot(1, len(numerical_cols), i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

for col in [
    "beer_servings",
    "spirit_servings",
    "wine_servings",
    "total_litres_of_pure_alcohol",
]:
    df[col] = df[col].fillna(df[col].median())

df = pd.get_dummies(df, columns=["continent"], dummy_na=False, drop_first=True)
df = df.drop(["country", "Unnamed: 0"], axis=1)
df = df.drop_duplicates()

print(df.head())
print(df.info())

from sklearn.model_selection import train_test_split

X = df.drop("total_litres_of_pure_alcohol", axis=1)
y = df["total_litres_of_pure_alcohol"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=df["continent_Asia"]
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=X_temp["continent_Asia"]
)


from sklearn.preprocessing import StandardScaler

X_train["beer_spirit_interaction"] = (
    X_train["beer_servings"] * X_train["spirit_servings"]
)
X_train["total_alcohol_servings"] = (
    X_train["beer_servings"] + X_train["spirit_servings"] + X_train["wine_servings"]
)

X_test["beer_spirit_interaction"] = X_test["beer_servings"] * X_test["spirit_servings"]
X_test["total_alcohol_servings"] = (
    X_test["beer_servings"] + X_test["spirit_servings"] + X_test["wine_servings"]
)

X_val["beer_spirit_interaction"] = X_val["beer_servings"] * X_val["spirit_servings"]
X_val["total_alcohol_servings"] = (
    X_val["beer_servings"] + X_val["spirit_servings"] + X_val["wine_servings"]
)


numerical_cols = [
    "beer_servings",
    "spirit_servings",
    "wine_servings",
    "beer_spirit_interaction",
    "total_alcohol_servings",
]
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])

print(X_train.head())
print(X_test.head())
print(X_val.head())

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


linear_regressor = LinearRegression()
random_forest_regressor = RandomForestRegressor()


linear_regressor.fit(X_train, y_train)
random_forest_regressor.fit(X_train, y_train)


linear_predictions = linear_regressor.predict(X_val)
random_forest_predictions = random_forest_regressor.predict(X_val)


linear_r2 = r2_score(y_val, linear_predictions)
random_forest_r2 = r2_score(y_val, random_forest_predictions)

print(f"Linear Regression R-squared: {linear_r2}")
print(f"Random Forest Regressor R-squared: {random_forest_r2}")

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

param_grid_linear = {
    "fit_intercept": [True, False],
}
linear_regressor_tuned = LinearRegression()
grid_search_linear = GridSearchCV(
    linear_regressor_tuned, param_grid_linear, scoring="r2", cv=5
)
grid_search_linear.fit(X_train, y_train)

linear_regressor = grid_search_linear.best_estimator_
linear_regressor.fit(X_train, y_train)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
random_forest_regressor_tuned = RandomForestRegressor(random_state=42)
random_search_rf = RandomizedSearchCV(
    random_forest_regressor_tuned,
    param_grid_rf,
    n_iter=10,
    scoring="r2",
    cv=5,
    random_state=42,
)
random_search_rf.fit(X_train, y_train)


random_forest_regressor = random_search_rf.best_estimator_
random_forest_regressor.fit(X_train, y_train)


linear_predictions = linear_regressor.predict(X_val)
random_forest_predictions = random_forest_regressor.predict(X_val)
linear_r2_tuned = r2_score(y_val, linear_predictions)
random_forest_r2_tuned = r2_score(y_val, random_forest_predictions)

print(f"Tuned Linear Regression R-squared: {linear_r2_tuned}")
print(f"Tuned Random Forest Regressor R-squared: {random_forest_r2_tuned}")

best_model = (
    linear_regressor
    if linear_r2_tuned > random_forest_r2_tuned
    else random_forest_regressor
)
best_r2 = max(linear_r2_tuned, random_forest_r2_tuned)
print(f"Best Model: {type(best_model).__name__}, R-squared: {best_r2}")

from sklearn.metrics import r2_score

best_model_predictions = best_model.predict(X_test)
best_model_r2 = r2_score(y_test, best_model_predictions)

print(f"Best Model R-squared on Test Set: {best_model_r2}")

# Data Visualization

images_dir = "static/images"
os.makedirs(images_dir, exist_ok=True)

plt.figure(figsize=(15, 10))
for i, col in enumerate(
    [
        "beer_servings",
        "spirit_servings",
        "wine_servings",
        "total_litres_of_pure_alcohol",
    ]
):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "histograms.png"))
plt.close()

plt.figure(figsize=(10, 8))
sns.pairplot(
    df,
    vars=[
        "beer_servings",
        "spirit_servings",
        "wine_servings",
        "total_litres_of_pure_alcohol",
    ],
    hue="continent_Asia",
    palette="viridis",
)
plt.savefig(os.path.join(images_dir, "scatter_matrix.png"))
plt.close()


plt.figure(figsize=(12, 6))
continent_cols = [
    "continent_Asia",
    "continent_Europe",
    "continent_North America",
    "continent_Oceania",
    "continent_South America",
]
continent_means = df.groupby(continent_cols)[["total_litres_of_pure_alcohol"]].mean()
continent_means.plot(kind="bar")
plt.title("Average Alcohol Consumption by Continent")
plt.xlabel("Continent")
plt.ylabel("Average Total Litres of Pure Alcohol")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "alcohol_by_continent.png"))
plt.close()


def preprocess_new_data(
    data: dict, scaler: StandardScaler, continent_columns: list
) -> pd.DataFrame:
    new_df = pd.DataFrame([data])

    for col in continent_columns:
        new_df[col] = 0

    if f'continent_{data["continent"]}' in continent_columns:
        new_df[f'continent_{data["continent"]}'] = 1

    new_df = new_df.drop("continent", axis=1)

    new_df["beer_spirit_interaction"] = (
        new_df["beer_servings"] * new_df["spirit_servings"]
    )
    new_df["total_alcohol_servings"] = (
        new_df["beer_servings"] + new_df["spirit_servings"] + new_df["wine_servings"]
    )

    numerical_cols = [
        "beer_servings",
        "spirit_servings",
        "wine_servings",
        "beer_spirit_interaction",
        "total_alcohol_servings",
    ]

    new_df[numerical_cols] = scaler.transform(new_df[numerical_cols])
    train_cols = X_train.columns
    new_df = new_df[train_cols]

    return new_df


continent_cols_from_train = [col for col in X_train.columns if "continent_" in col]


sample_data = {
    "beer_servings": 100,
    "spirit_servings": 50,
    "wine_servings": 20,
    "continent": "Europe",
}

preprocessed_sample_data = preprocess_new_data(
    sample_data, scaler, continent_cols_from_train
)
loaded_model = joblib.load("best_model.pkl")
prediction = loaded_model.predict(preprocessed_sample_data)

print(f"Sample Data: {sample_data}")
print(f"Preprocessed Sample Data Head:\n{preprocessed_sample_data.head()}")
print(f"Predicted total_litres_of_pure_alcohol: {prediction[0]}")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

continent_cols = [
    "continent_Asia",
    "continent_Europe",
    "continent_North America",
    "continent_Oceania",
    "continent_South America",
]


def preprocess_input(data, scaler, continent_columns):
    df = pd.DataFrame([data])
    for col in continent_columns:
        df[col] = 0
    continent_col = f"continent_{data['continent']}"
    if continent_col in continent_columns:
        df[continent_col] = 1
    df.drop("continent", axis=1, inplace=True)
    df["beer_spirit_interaction"] = df["beer_servings"] * df["spirit_servings"]
    df["total_alcohol_servings"] = (
        df["beer_servings"] + df["spirit_servings"] + df["wine_servings"]
    )
    num_cols = [
        "beer_servings",
        "spirit_servings",
        "wine_servings",
        "beer_spirit_interaction",
        "total_alcohol_servings",
    ]
    df[num_cols] = scaler.transform(df[num_cols])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


st.title("🍺 Alcohol Consumption Predictor")
st.markdown("### Predict Total Litres of Pure Alcohol Consumed per Person")

continent = st.selectbox(
    "Continent", ["Asia", "Europe", "North America", "Oceania", "South America"]
)
beer_servings = st.slider("Beer Servings", 0, 500, 100)
spirit_servings = st.slider("Spirit Servings", 0, 500, 50)
wine_servings = st.slider("Wine Servings", 0, 500, 30)

input_data = {
    "beer_servings": beer_servings,
    "spirit_servings": spirit_servings,
    "wine_servings": wine_servings,
    "continent": continent,
}

if st.button("Predict"):
    processed_input = preprocess_input(input_data, scaler, continent_cols)
    prediction = model.predict(processed_input)[0]
    st.success(f"Predicted total litres of pure alcohol: **{prediction:.2f} L/person**")

st.markdown("### 📊 Data Insights")
st.image("static/images/histograms.png", caption="Distributions")
st.image("static/images/scatter_matrix.png", caption="Scatter Matrix")
st.image("static/images/alcohol_by_continent.png", caption="Alcohol by Continent")
