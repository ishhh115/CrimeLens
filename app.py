from flask import Flask, jsonify, render_template
from analysis import load_data, state_wise_crimes, year_wise_trend, crime_type_breakdown, crimes_against_women
from model import predict_crimes, get_all_states, get_risk_scores

app = Flask(__name__)
df = load_data()

@app.route('/')
def index():
    return render_template('index.html')

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
    result = predict_crimes(df, state.upper())
    if result is None:
        return jsonify({'error': 'Not enough data'}), 400
    return jsonify(result)

@app.route('/api/states')
def api_states():
    return jsonify(get_all_states(df))

@app.route('/api/risk-scores')
def api_risk_scores():
    return jsonify(get_risk_scores(df))

if __name__ == '__main__':
    app.run(debug=True)