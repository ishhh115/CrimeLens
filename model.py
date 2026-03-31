import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def build_features(df):
    # Group by state and year
    state_year = df.groupby(['STATE/UT', 'YEAR']).agg(
        total_crimes=('TOTAL IPC CRIMES', 'sum'),
        murder=('MURDER', 'sum'),
        rape=('RAPE', 'sum'),
        theft=('THEFT', 'sum'),
        robbery=('ROBBERY', 'sum'),
        riots=('RIOTS', 'sum')
    ).reset_index()

    # Feature 1: violent crime ratio
    state_year['violent_ratio'] = (
        (state_year['murder'] + state_year['rape'] + state_year['robbery']) /
        state_year['total_crimes']
    ).fillna(0)

    # Feature 2: year on year change
    state_year = state_year.sort_values(['STATE/UT', 'YEAR'])
    state_year['yoy_change'] = state_year.groupby('STATE/UT')['total_crimes'].pct_change().fillna(0)

    # Feature 3: normalize total crimes to 0-100
    max_c = state_year['total_crimes'].max()
    min_c = state_year['total_crimes'].min()
    state_year['crime_score'] = ((state_year['total_crimes'] - min_c) / (max_c - min_c) * 100)

    # Create risk label based on crime_score
    def get_risk(score):
        if score <= 25: return 'Low'
        elif score <= 50: return 'Medium'
        elif score <= 75: return 'High'
        else: return 'Critical'

    state_year['risk_level'] = state_year['crime_score'].apply(get_risk)
    return state_year

def train_model(df):
    features = build_features(df)

    X = features[['crime_score', 'violent_ratio', 'yoy_change', 'theft', 'murder']]
    y = features['risk_level']

    # Encode labels (Low/Medium/High/Critical → 0/1/2/3)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    return model, le, features

def predict_risk(df, state):
    model, le, features = train_model(df)

    state_data = features[features['STATE/UT'] == state]
    if state_data.empty:
        return None

    latest = state_data.sort_values('YEAR').iloc[-1]

    X_pred = pd.DataFrame([{
        'crime_score': latest['crime_score'],
        'violent_ratio': latest['violent_ratio'],
        'yoy_change': latest['yoy_change'],
        'theft': latest['theft'],
        'murder': latest['murder']
    }])

    pred_encoded = model.predict(X_pred)[0]
    pred_proba = model.predict_proba(X_pred)[0]
    risk_level = le.inverse_transform([pred_encoded])[0]
    confidence = round(max(pred_proba) * 100, 1)

    historical = state_data[['YEAR', 'total_crimes', 'risk_level']].to_dict(orient='records')

    return {
        'state': state,
        'predicted_risk': risk_level,
        'confidence': confidence,
        'historical': historical,
        'feature_importance': {
            'crime_score': round(latest['crime_score'], 1),
            'violent_ratio': round(latest['violent_ratio'], 3),
            'yoy_change': round(latest['yoy_change'], 3)
        }
    }

def get_all_states(df):
    return sorted(df['STATE/UT'].unique().tolist())