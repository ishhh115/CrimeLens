import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Global cache — train once, reuse forever
_cached_model = None
_cached_le = None
_cached_features = None

def build_features(df):
    state_year = df.groupby(['STATE/UT', 'YEAR']).agg(
        total_crimes=('TOTAL IPC CRIMES', 'sum'),
        murder=('MURDER', 'sum'),
        rape=('RAPE', 'sum'),
        theft=('THEFT', 'sum'),
        robbery=('ROBBERY', 'sum'),
        riots=('RIOTS', 'sum')
    ).reset_index()

    state_year['violent_ratio'] = (
        (state_year['murder'] + state_year['rape'] + state_year['robbery']) /
        state_year['total_crimes']
    ).fillna(0)

    state_year = state_year.sort_values(['STATE/UT', 'YEAR'])
    state_year['yoy_change'] = state_year.groupby('STATE/UT')['total_crimes'].pct_change().fillna(0)

    max_c = state_year['total_crimes'].max()
    min_c = state_year['total_crimes'].min()
    state_year['crime_score'] = ((state_year['total_crimes'] - min_c) / (max_c - min_c) * 100)

    def get_risk(score):
        if score <= 25: return 'Low'
        elif score <= 50: return 'Medium'
        elif score <= 75: return 'High'
        else: return 'Critical'

    state_year['risk_level'] = state_year['crime_score'].apply(get_risk)
    return state_year

def train_model(df):
    features = build_features(df)

    X = features[['violent_ratio', 'yoy_change', 'theft', 'murder', 'rape', 'robbery']]
    y = features['risk_level']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    return model, le, features

def predict_risk(df, state):
    global _cached_model, _cached_le, _cached_features
    if _cached_model is None:
        _cached_model, _cached_le, _cached_features = train_model(df)
    model, le, features = _cached_model, _cached_le, _cached_features

    state_data = features[features['STATE/UT'] == state]
    if state_data.empty:
        return None

    latest = state_data.sort_values('YEAR').iloc[-1]

    X_pred = pd.DataFrame([{
        'violent_ratio': latest['violent_ratio'],
        'yoy_change': latest['yoy_change'],
        'theft': latest['theft'],
        'murder': latest['murder'],
        'rape': latest['rape'],
        'robbery': latest['robbery']
    }])

    pred_encoded = model.predict(X_pred)[0]
    pred_proba = model.predict_proba(X_pred)[0]
    risk_level = le.inverse_transform([pred_encoded])[0]
    confidence = round(max(pred_proba) * 100, 1)

    historical = state_data[['YEAR', 'total_crimes', 'risk_level']].to_dict(orient='records')

    feature_names = ['violent_ratio', 'yoy_change', 'theft', 'murder', 'rape', 'robbery']
    importances = model.feature_importances_

    feature_impact = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    def explain_feature(name, value, importance):
        explanations = {
            'violent_ratio': f"Violent crimes make up {round(value*100, 1)}% of all crimes — {'concerning' if value > 0.1 else 'within normal range'}",
            'yoy_change': f"Crime {'increased' if value > 0 else 'decreased'} by {round(abs(value)*100, 1)}% compared to previous year",
            'theft': f"Theft cases: {int(value):,} — {'dominant crime type' if value > 50000 else 'significant contributor'}",
            'murder': f"Murder cases: {int(value):,} — {'high' if value > 2000 else 'moderate' if value > 500 else 'relatively low'}",
            'rape': f"Rape cases: {int(value):,} — {'high concern' if value > 2000 else 'requires monitoring'}",
            'robbery': f"Robbery cases: {int(value):,} — {'elevated' if value > 1000 else 'moderate'}"
        }
        return {
            'feature': name,
            'importance': round(float(importance) * 100, 1),
            'explanation': explanations.get(name, name)
        }

    top3_reasons = [
        explain_feature(name, latest[name], imp)
        for name, imp in feature_impact[:3]
    ]

    return {
        'state': state,
        'predicted_risk': risk_level,
        'confidence': confidence,
        'historical': historical,
        'feature_importance': {
            'crime_score': round(latest['crime_score'], 1),
            'violent_ratio': round(latest['violent_ratio'], 3),
            'yoy_change': round(latest['yoy_change'], 3)
        },
        'why_prediction': top3_reasons
    }

def get_all_states(df):
    return sorted(df['STATE/UT'].unique().tolist())

def get_model_accuracy(df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    features = build_features(df)

    X = features[['violent_ratio', 'yoy_change', 'theft', 'murder', 'rape', 'robbery']]
    y = features['risk_level']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)

    return {'accuracy': accuracy}