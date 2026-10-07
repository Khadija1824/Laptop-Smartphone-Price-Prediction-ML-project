import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# 1. Load the dataset
df = pd.read_csv('laptop_data.csv')

# 2. Define Features (X) and Target (y)
X = df.drop(columns=['Price'])
y = df['Price']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing categorical columns
categorical_features = ['Company', 'TypeName', 'Memory']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 5. Build Pipeline with Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. Train Model
model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# 8. Save Pipeline to disk
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully as model.pkl!")