import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("heart_disease_uci.csv")

#Sci-kit pipeline
X = df.drop("num", axis=1)
y = df["num"]
X = X.drop(["id", "dataset"], axis=1)

cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())])
cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)])

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,
stratify=y)

#KNN pipeline
knn_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier())
    ]
)
knn_pipeline.fit(X_train, y_train)

#Saving the model
import joblib
joblib.dump(knn_pipeline,"heart_disease_knn_pipeline.pkl")
print("Model saved")