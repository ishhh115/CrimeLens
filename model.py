import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from analysis import get_state_features

def assign_risk_label(crime_rate):
    if crime_rate < 3000:
        return 'Low'
    elif crime_rate < 6000:
        return 'Medium'
    elif crime_rate < 10000:
        return 'High'
    else:
        return 'Critical'

def train_and_save_model(df):
    features = get_state_features(df)
    features['risk_label'] = features['crime_rate_per_lakh'].apply(assign_risk_label)

    X = features[['crime_rate_per_lakh', 'avg_yoy_change', 'total_crimes']]
    y = features['risk_label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    joblib.dump(model, 'crime_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Model trained and saved!")
    return model, le, features

def load_model():
    if os.path.exists('crime_model.pkl'):
        model = joblib.load('crime_model.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le
    return None, None

def get_risk_scores(df):
    model, le = load_model()
    if model is None:
        train_and_save_model(df)
        model, le = load_model()

    features = get_state_features(df)
    X = features[['crime_rate_per_lakh', 'avg_yoy_change', 'total_crimes']]

    predictions = model.predict(X)
    risk_labels = le.inverse_transform(predictions)

    features['risk_level'] = risk_labels
    features['risk_label_manual'] = features['crime_rate_per_lakh'].apply(assign_risk_label)

    result = features[['STATE/UT', 'crime_rate_per_lakh', 'avg_yoy_change', 'risk_level']].copy()
    result = result.sort_values('crime_rate_per_lakh', ascending=False)
    return result.to_dict(orient='records')

def predict_crimes(df, state):
    state_df = df[df['STATE/UT'] == state]
    yearly = state_df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().reset_index()

    if len(yearly) < 2:
        return None

    from sklearn.linear_model import LinearRegression
    X = yearly['YEAR'].values.reshape(-1, 1)
    y = yearly['TOTAL IPC CRIMES'].values
    reg = LinearRegression()
    reg.fit(X, y)

    future_years = np.array([[2015], [2016], [2017]])
    predictions = reg.predict(future_years)

    result = []
    for year, pred in zip(future_years.flatten(), predictions):
        result.append({
            'year': int(year),
            'predicted_crimes': int(pred)
        })

    return {
        'state': state,
        'historical': yearly.to_dict(orient='records'),
        'predictions': result
    }

def get_all_states(df):
    return sorted(df['STATE/UT'].unique().tolist())