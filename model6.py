import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Step 1: Data Preprocessing
def clean_data(input_file, output_file):
    rows = []
    cols_expected = 13
    with open(input_file, "r") as file:
        for line in file:
            fields = line.strip().split(",")
            if len(fields) == cols_expected:
                rows.append(fields)
    with open(output_file, "w") as file:
        for row in rows:
            file.write(",".join(row) + "\n")

def preprocess_data(file):
    cols = [
        "survival", "alive", "age", "effusion",
        "shortening", "epss", "lvdd", "motion_score",
        "motion_index", "extra", "name", "group", "alive_1yr"
    ]
    data = pd.read_csv(file, header=None, names=cols, na_values="?")
    data.drop(columns=["extra", "name", "group"], inplace=True)
    data.fillna(data.median(), inplace=True)
    scaler = MinMaxScaler()
    numeric_cols = ["shortening", "epss", "lvdd", "motion_score", "motion_index"]
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Step 2: Assign Emotion Labels
def assign_emotions(df):
    conditions = [
        (df["shortening"] < 0.3) & (df["motion_score"] > 0.7),
        (df["shortening"] >= 0.3) & (df["shortening"] < 0.5) & (df["motion_score"] > 0.5),
        (df["shortening"] >= 0.5) & (df["shortening"] < 0.7) & (df["motion_score"] <= 0.5),
        (df["shortening"] >= 0.7) & (df["motion_score"] <= 0.6),
        (df["shortening"] < 0.3) & (df["motion_score"] <= 0.4),
    ]
    emotions = ["Anxious", "Tense", "Peaceful", "Energized", "Calm"]
    df["emotion"] = np.select(conditions, emotions, default="Unknown")
    return df

# Step 3: Add Noise
def add_noise(df, level=0.05):
    noisy_df = df.copy()
    for col in ["shortening", "motion_score"]:
        noise = level * np.random.normal(size=df[col].shape)
        noisy_df[col] = df[col] + noise
        noisy_df[col] = noisy_df[col].clip(0, 1)
    return noisy_df

# Step 4: Train Models
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Dynamically determine the labels in y_test
    unique_labels = np.unique(y_test)
    emotion_names = ["Anxious", "Tense", "Peaceful", "Energized", "Calm"]
    target_names = [emotion_names[label] for label in unique_labels]

    # Print the classification report
    print(classification_report(y_test, y_pred, target_names=target_names))
    return model

# Main Execution
input_file = "echocardiogram.data"
output_file = "cleaned_data.csv"
clean_data(input_file, output_file)
data = preprocess_data(output_file)
data = assign_emotions(data)
data = add_noise(data)

# Prepare Data for Training
data = data[data["emotion"] != "Unknown"].copy()
data["emotion"] = data["emotion"].astype("category").cat.codes
X = data[["shortening", "motion_score"]]
y = data["emotion"]

# Handle Imbalance
min_class_samples = y.value_counts().min()
k_neighbors = min(5, min_class_samples - 1)  # Ensure k_neighbors < number of samples in smallest class

if min_class_samples > 1 and k_neighbors >= 1:  # Proceed only if enough samples for SMOTE
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X, y = smote.fit_resample(X, y)
else:
    print("Not enough samples for SMOTE. Continuing without oversampling.")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
print("Random Forest Results:")
rf = RandomForestClassifier(random_state=42)
train_model(rf, X_train, X_test, y_train, y_test)

print("Logistic Regression Results:")
lr = LogisticRegression(random_state=42, max_iter=1000)
train_model(lr, X_train, X_test, y_train, y_test)

print("SVM Results:")
svm = SVC(kernel='rbf', random_state=42)
train_model(svm, X_train, X_test, y_train, y_test)

print("k-NN Results:")
knn = KNeighborsClassifier(n_neighbors=5)
train_model(knn, X_train, X_test, y_train, y_test)
