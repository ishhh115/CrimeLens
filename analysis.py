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