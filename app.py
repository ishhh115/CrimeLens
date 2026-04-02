from flask import Flask, jsonify, render_template
from analysis import load_data, state_wise_crimes, year_wise_trend, crime_type_breakdown, crimes_against_women
from model import predict_risk, get_all_states

app = Flask(__name__)
df = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/api/state-crimes')
def api_state_crimes():
    return jsonify(state_wise_crimes(df))

@app.route('/api/year-trend')
def api_year_trend():
    return jsonify(year_wise_trend(df))

@app.route('/api/crime-types')
def api_crime_types():
    return jsonify(crime_type_breakdown(df))

@app.route('/api/women-crimes')
def api_women_crimes():
    return jsonify(crimes_against_women(df))

@app.route('/api/predict/<state>')
def api_predict(state):
    result = predict_risk(df, state.upper())
    if result is None:
        return jsonify({'error': 'Not enough data'}), 400
    return jsonify(result)

@app.route('/api/states')
def api_states():
    return jsonify(get_all_states(df))

@app.route('/api/risk-scores')
def api_risk_scores():
    from analysis import state_risk_scores
    return jsonify(state_risk_scores(df))

@app.route('/api/anomalies')
def api_anomalies():
    from analysis import detect_anomalies
    return jsonify(detect_anomalies(df))

@app.route('/api/policy-insights')
def api_policy_insights():
    from analysis import policy_insights
    return jsonify(policy_insights(df))

@app.route('/api/accuracy')
def api_accuracy():
    from model import get_model_accuracy
    return jsonify(get_model_accuracy(df))

@app.route('/api/choropleth')
def api_choropleth():
    from analysis import generate_choropleth
    return generate_choropleth(df)

@app.route('/map')
def map_page():
    return render_template('map.html')

if __name__ == '__main__':
    app.run(debug=True)