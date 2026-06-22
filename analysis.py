import pandas as pd
import numpy as np

STATE_POPULATION = {
    'ANDHRA PRADESH': 49386799, 'ARUNACHAL PRADESH': 1383727,
    'ASSAM': 31205576, 'BIHAR': 104099452, 'CHHATTISGARH': 25545198,
    'GOA': 1458545, 'GUJARAT': 60439692, 'HARYANA': 25351462,
    'HIMACHAL PRADESH': 6864602, 'JAMMU & KASHMIR': 12541302,
    'JHARKHAND': 32988134, 'KARNATAKA': 61095297, 'KERALA': 33406061,
    'MADHYA PRADESH': 72626809, 'MAHARASHTRA': 112374333,
    'MANIPUR': 2855794, 'MEGHALAYA': 2966889, 'MIZORAM': 1097206,
    'NAGALAND': 1978502, 'ODISHA': 41974218, 'PUNJAB': 27743338,
    'RAJASTHAN': 68548437, 'SIKKIM': 610577, 'TAMIL NADU': 72147030,
    'TRIPURA': 3673917, 'UTTAR PRADESH': 199812341,
    'UTTARAKHAND': 10086292, 'WEST BENGAL': 91276115,
    'DELHI UT': 16787941, 'CHANDIGARH': 1055450,
    'PUDUCHERRY': 1247953, 'A & N ISLANDS': 380581,
    'D & N HAVELI': 343709, 'DAMAN & DIU': 243247, 'LAKSHADWEEP': 64473,
}

def load_data():
    df1 = pd.read_csv('data/crime_data.csv')
    df2 = pd.read_csv('data/ipc_2013.csv')
    df3 = pd.read_csv('data/ipc_2014.csv')
    
    # Align 2014 columns to the standard format to avoid silent dropping
    df3 = df3.rename(columns={
        'States/UTs': 'STATE/UT',
        'District': 'DISTRICT',
        'Year': 'YEAR',
        'Murder': 'MURDER',
        'Rape': 'RAPE',
        'Theft': 'THEFT',
        'Robbery': 'ROBBERY',
        'Riots': 'RIOTS',
        'Cheating': 'CHEATING',
        'Arson': 'ARSON',
        'Dowry Deaths': 'DOWRY DEATHS',
        'Total Cognizable IPC crimes': 'TOTAL IPC CRIMES',
        'Kidnapping & Abduction_Total': 'KIDNAPPING & ABDUCTION',
        'Kidnapping & Abduction of Women to compel her for marriage': 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
        'Assault on Women with intent to outrage her Modesty': 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
        'Cruelty by Husband or his Relatives': 'CRUELTY BY HUSBAND OR HIS RELATIVES',
        'Importation of Girls from Foreign Country': 'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
        'Causing Death by Negligence': 'CAUSING DEATH BY NEGLIGENCE',
        'Insult to the Modesty of Women': 'INSULT TO MODESTY OF WOMEN'
    })
    
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['STATE/UT'])
    df['STATE/UT'] = df['STATE/UT'].str.strip().str.upper()
    return df

def get_state_features(df):
    state_group = df.groupby('STATE/UT').agg(
        total_crimes=('TOTAL IPC CRIMES', 'sum'),
        total_cases=('TOTAL IPC CRIMES', 'count'),
    ).reset_index()
    state_group['population'] = state_group['STATE/UT'].map(STATE_POPULATION)
    state_group = state_group.dropna(subset=['population'])
    state_group['crime_rate_per_lakh'] = (
        state_group['total_crimes'] / state_group['population'] * 100000
    ).round(2)
    yearly = df.groupby(['STATE/UT', 'YEAR'])['TOTAL IPC CRIMES'].sum().reset_index()
    yearly = yearly.sort_values(['STATE/UT', 'YEAR'])
    yearly['yoy_change'] = yearly.groupby('STATE/UT')['TOTAL IPC CRIMES'].pct_change() * 100
    avg_yoy = yearly.groupby('STATE/UT')['yoy_change'].mean().reset_index()
    avg_yoy.columns = ['STATE/UT', 'avg_yoy_change']
    state_group = state_group.merge(avg_yoy, on='STATE/UT', how='left')
    state_group['avg_yoy_change'] = state_group['avg_yoy_change'].fillna(0).round(2)
    return state_group

def state_wise_crimes(df):
    state_data = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
    state_data = state_data.sort_values('TOTAL IPC CRIMES', ascending=False)
    return state_data.to_dict(orient='records')

def year_wise_trend(df):
    year_data = df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().reset_index()
    year_data = year_data.sort_values('YEAR')
    return year_data.to_dict(orient='records')

def crime_type_breakdown(df):
    crime_cols = [
        'MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION',
        'ROBBERY', 'BURGLARY', 'THEFT', 'RIOTS',
        'DOWRY DEATHS', 'ARSON', 'CHEATING'
    ]
    breakdown = df[crime_cols].sum().reset_index()
    breakdown.columns = ['crime_type', 'count']
    breakdown = breakdown.sort_values('count', ascending=False)
    return breakdown.to_dict(orient='records')

def crimes_against_women(df):
    women_cols = [
        'RAPE',
        'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
        'DOWRY DEATHS',
        'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
        'INSULT TO MODESTY OF WOMEN',
        'CRUELTY BY HUSBAND OR HIS RELATIVES'
    ]
    df = df.copy()
    df['CRIMES_AGAINST_WOMEN'] = df[women_cols].sum(axis=1)
    caw = df.groupby('YEAR')['CRIMES_AGAINST_WOMEN'].sum().reset_index()
    return caw.to_dict(orient='records')

def state_risk_scores(df):
    # Get total crimes per state
    state_data = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
    
    # Normalize to 0-100 score
    max_crimes = state_data['TOTAL IPC CRIMES'].max()
    min_crimes = state_data['TOTAL IPC CRIMES'].min()
    
    state_data['risk_score'] = ((state_data['TOTAL IPC CRIMES'] - min_crimes) / 
                                 (max_crimes - min_crimes) * 100).round(1)
    
    # Assign risk level based on score
    def get_risk_level(score):
        if score <= 25:
            return 'Low'
        elif score <= 50:
            return 'Medium'
        elif score <= 75:
            return 'High'
        else:
            return 'Critical'
    
    state_data['risk_level'] = state_data['risk_score'].apply(get_risk_level)
    state_data = state_data.sort_values('risk_score', ascending=False)
    return state_data.to_dict(orient='records')

def detect_anomalies(df):
    # Group by state and year
    state_year = df.groupby(['STATE/UT', 'YEAR'])['TOTAL IPC CRIMES'].sum().reset_index()
    
    anomalies = []
    
    for state in state_year['STATE/UT'].unique():
        state_data = state_year[state_year['STATE/UT'] == state].sort_values('YEAR')
        
        mean = state_data['TOTAL IPC CRIMES'].mean()
        std = state_data['TOTAL IPC CRIMES'].std()
        
        if std == 0:
            continue
            
        for _, row in state_data.iterrows():
            z_score = (row['TOTAL IPC CRIMES'] - mean) / std
            # z_score > 2 = unusually high, < -2 = unusually low
            if abs(z_score) > 1.5:
                anomalies.append({
                    'state': row['STATE/UT'],
                    'year': int(row['YEAR']),
                    'total_crimes': int(row['TOTAL IPC CRIMES']),
                    'z_score': round(float(z_score), 2),
                    'type': 'unusually high' if z_score > 0 else 'unusually low',
                    'message': f"{int(row['YEAR'])} was {('unusually high' if z_score > 0 else 'unusually low')} vs historical trend (z={round(float(z_score),2)})"
                })
    
    return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)

def policy_insights(df):
    crime_cols = [
        'MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION',
        'ROBBERY', 'BURGLARY', 'THEFT', 'RIOTS',
        'DOWRY DEATHS', 'ARSON', 'CHEATING'
    ]
    
    state_data = df.groupby('STATE/UT')[crime_cols + ['TOTAL IPC CRIMES']].sum()
    
    insights = []
    for state, row in state_data.iterrows():
        total = row['TOTAL IPC CRIMES']
        if total == 0:
            continue
        
        # Find dominant crime type
        crime_counts = {col: row[col] for col in crime_cols}
        dominant_crime = max(crime_counts, key=crime_counts.get)
        dominant_pct = round(crime_counts[dominant_crime] / total * 100, 1)
        
        # Find fastest growing concern
        if dominant_crime in ['RAPE', 'DOWRY DEATHS', 'KIDNAPPING & ABDUCTION']:
            focus = f"Priority: Women safety — {dominant_crime.title()} accounts for {dominant_pct}% of crimes"
        elif dominant_crime == 'THEFT':
            focus = f"Focus: Theft reduction — accounts for {dominant_pct}% of all crimes"
        elif dominant_crime == 'MURDER':
            focus = f"Alert: Violent crime — Murder is dominant at {dominant_pct}% of crimes"
        elif dominant_crime == 'CHEATING':
            focus = f"Focus: Economic crime — Cheating/fraud accounts for {dominant_pct}% of crimes"
        else:
            focus = f"Focus: {dominant_crime.title()} reduction — accounts for {dominant_pct}% of crimes"
        
        insights.append({
            'state': state,
            'dominant_crime': dominant_crime,
            'dominant_pct': dominant_pct,
            'policy': focus
        })
    
    return sorted(insights, key=lambda x: x['dominant_pct'], reverse=True)

def generate_choropleth(df):
    import folium

    state_data = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
    max_c = state_data['TOTAL IPC CRIMES'].max()
    min_c = state_data['TOTAL IPC CRIMES'].min()
    state_data['risk_score'] = ((state_data['TOTAL IPC CRIMES'] - min_c) / (max_c - min_c) * 100).round(1)

    # Fix state names to match GeoJSON
    name_map = {
        'ANDHRA PRADESH': 'Andhra Pradesh',
        'ARUNACHAL PRADESH': 'Arunachal Pradesh',
        'ASSAM': 'Assam',
        'BIHAR': 'Bihar',
        'CHHATTISGARH': 'Chhattisgarh',
        'GOA': 'Goa',
        'GUJARAT': 'Gujarat',
        'HARYANA': 'Haryana',
        'HIMACHAL PRADESH': 'Himachal Pradesh',
        'JAMMU & KASHMIR': 'Jammu and Kashmir',
        'JHARKHAND': 'Jharkhand',
        'KARNATAKA': 'Karnataka',
        'KERALA': 'Kerala',
        'MADHYA PRADESH': 'Madhya Pradesh',
        'MAHARASHTRA': 'Maharashtra',
        'MANIPUR': 'Manipur',
        'MEGHALAYA': 'Meghalaya',
        'MIZORAM': 'Mizoram',
        'NAGALAND': 'Nagaland',
        'ODISHA': 'Odisha',
        'PUNJAB': 'Punjab',
        'RAJASTHAN': 'Rajasthan',
        'SIKKIM': 'Sikkim',
        'TAMIL NADU': 'Tamil Nadu',
        'TRIPURA': 'Tripura',
        'UTTAR PRADESH': 'Uttar Pradesh',
        'UTTARAKHAND': 'Uttarakhand',
        'WEST BENGAL': 'West Bengal',
        'DELHI UT': 'NCT of Delhi',
    }

    state_data['state_name'] = state_data['STATE/UT'].map(name_map)
    state_data = state_data.dropna(subset=['state_name'])

    geojson_url = 'data/india_states.geojson'

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4, tiles='CartoDB positron')

    folium.Choropleth(
        geo_data=geojson_url,
        name='Crime Risk',
        data=state_data,
        columns=['state_name', 'risk_score'],
        key_on='feature.properties.NAME_1',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name='Crime Risk Score (0-100)',
        nan_fill_color='#f0f0f0'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()

def state_action_brief(df, state):
    from model import predict_risk
    
    # Get prediction
    prediction = predict_risk(df, state)
    if prediction is None:
        return None
    
    # Get anomalies for this state
    anomalies = detect_anomalies(df)
    state_anomalies = [a for a in anomalies if a['state'] == state]
    
    # Get policy insight for this state
    insights = policy_insights(df)
    state_insight = next((i for i in insights if i['state'] == state), None)
    
    # Get crimes against women trend
    state_df = df[df['STATE/UT'] == state]
    women_cols = [
        'RAPE', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
        'DOWRY DEATHS', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
        'CRUELTY BY HUSBAND OR HIS RELATIVES'
    ]
    women_total = state_df[women_cols].sum().sum()
    total_crimes = state_df['TOTAL IPC CRIMES'].sum()
    women_pct = round(women_total / total_crimes * 100, 1) if total_crimes > 0 else 0

    # Generate 90-day outlook
    yoy = prediction['feature_importance']['yoy_change']
    if yoy > 0.05:
        outlook = "Rising — intervention recommended within 30 days"
    elif yoy < -0.05:
        outlook = "Improving — continue current strategy"
    else:
        outlook = "Stable — monitor quarterly"

    # Generate interventions
    interventions = []
    if state_insight:
        interventions.append(f"Priority: Reduce {state_insight['dominant_crime'].title()} — accounts for {state_insight['dominant_pct']}% of crimes")
    if women_pct > 15:
        interventions.append(f"Women safety alert: {women_pct}% of crimes target women — fast-track women safety units")
    if state_anomalies:
        latest_anomaly = state_anomalies[0]
        interventions.append(f"Anomaly: {latest_anomaly['message']} — investigate root cause")
    interventions.append("Deploy predictive patrolling in high-density crime districts")

    return {
        'state': state,
        'risk_level': prediction['predicted_risk'],
        'confidence': prediction['confidence'],
        'outlook': outlook,
        'top_drivers': prediction['why_prediction'],
        'anomaly_alert': state_anomalies[0]['message'] if state_anomalies else 'No anomalies detected',
        'interventions': interventions,
        'women_crime_pct': women_pct,
        'caveat': 'Based on NCRB data 2001-2012. Use alongside current ground intelligence.'
    }

def get_state_detailed_summary(df):
    # Risk scores
    risk_data = state_risk_scores(df)
    risk_df = pd.DataFrame(risk_data)
    
    # State features
    features_df = get_state_features(df)
    
    # Anomaly counts per state
    anomalies = detect_anomalies(df)
    anomaly_counts = {}
    for a in anomalies:
        state = a['state']
        anomaly_counts[state] = anomaly_counts.get(state, 0) + 1
        
    # Policy insights mapping
    insights = policy_insights(df)
    insights_map = {i['state']: i['policy'] for i in insights}
    dominant_crime_map = {i['state']: i['dominant_crime'] for i in insights}
    
    # Merge
    merged = risk_df.merge(features_df, on='STATE/UT', how='left')
    merged['anomaly_count'] = merged['STATE/UT'].map(anomaly_counts).fillna(0).astype(int)
    
    result = []
    for _, row in merged.iterrows():
        state_name = row['STATE/UT']
        result.append({
            'state': state_name,
            'total_crimes': int(row['TOTAL IPC CRIMES']),
            'risk_score': float(row['risk_score']),
            'risk_level': row['risk_level'],
            'population': int(row['population']) if not pd.isna(row['population']) else 0,
            'crime_rate': float(row['crime_rate_per_lakh']) if not pd.isna(row['crime_rate_per_lakh']) else 0.0,
            'yoy_change': float(row['avg_yoy_change']) if not pd.isna(row['avg_yoy_change']) else 0.0,
            'anomaly_count': int(row['anomaly_count']),
            'policy_brief': insights_map.get(state_name, 'No critical pattern flagged'),
            'dominant_crime': dominant_crime_map.get(state_name, 'None')
        })
    return result