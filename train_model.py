import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load the datasets
disease_df = pd.read_csv("disease_data_monthly.csv")
climate_df = pd.read_csv("climate_data_monthly.csv")
population_df = pd.read_csv("population_data.csv")
healthcare_df = pd.read_csv("healthcare_data.csv")

# Load the new disease data with symptoms, precautions, and cures
disease_info_df = pd.read_csv("disease_symptoms_precautions_with_cure.csv")

# Merge datasets based on disease
merged_df = disease_df.merge(climate_df, on=["Month", "State", "District"])
merged_df = merged_df.merge(population_df, on=["State", "District"])
merged_df = merged_df.merge(healthcare_df, on=["State", "District"])

# Merge the disease info data (symptoms, precautions, cures) into the main dataset
merged_df = merged_df.merge(disease_info_df[['Disease', 'Symptoms', 'Precautions', 'Cure']], 
                             left_on="Disease", right_on="Disease", how="left")

# Preprocessing text data (Symptoms, Precautions, Cure)
# Vectorize Symptoms, Precautions, and Cure columns using TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
symptoms_tfidf = vectorizer.fit_transform(merged_df['Symptoms'].fillna(''))
precautions_tfidf = vectorizer.fit_transform(merged_df['Precautions'].fillna(''))
cure_tfidf = vectorizer.fit_transform(merged_df['Cure'].fillna(''))

# Convert TF-IDF matrices to DataFrames and concatenate them to the features
symptoms_df = pd.DataFrame(symptoms_tfidf.toarray(), columns=[f'Symptom_{i}' for i in range(symptoms_tfidf.shape[1])])
precautions_df = pd.DataFrame(precautions_tfidf.toarray(), columns=[f'Precaution_{i}' for i in range(precautions_tfidf.shape[1])])
cure_df = pd.DataFrame(cure_tfidf.toarray(), columns=[f'Cure_{i}' for i in range(cure_tfidf.shape[1])])

# Concatenate these new features with the existing features
merged_df = pd.concat([merged_df, symptoms_df, precautions_df, cure_df], axis=1)

# Select features and target variable
features = [
    "AvgTemperature", "AvgHumidity", "TotalRainfall", "Population",
    "PopulationDensity", "UrbanizationRate", "Hospitals", "Beds", "Doctors", "Nurses"
] + list(symptoms_df.columns) + list(precautions_df.columns) + list(cure_df.columns)

target = "ReportedCases"

# Create a binary target variable: 1 for high outbreaks, 0 for low
merged_df["HighOutbreak"] = (merged_df[target] > 1000).astype(int)

# Split into features (X) and target (y)
X = merged_df[features]
y = merged_df["HighOutbreak"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Plot feature importances
feature_importances = model.feature_importances_
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Disease Outbreak Prediction")
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, "disease_outbreak_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")