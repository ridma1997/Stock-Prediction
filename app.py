from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go
import plotly
import json
from werkzeug.security import generate_password_hash, check_password_hash
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

logging.basicConfig(level=logging.INFO)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

def fetch_stock_data(ticker, interval):
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    return df

@app.route('/help')
def help():
    return render_template('help.html')

def predict_stock_prices(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Timestamp'] = df['Date'].map(pd.Timestamp.timestamp)
    X = df[['Timestamp']]
    y = df['Close']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_scaled)
    df['Predicted'] = predictions
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler, training_data_len, scaled_data, df

def build_lstm_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

def predict_future_prices(model, scaler, scaled_data, days=730):
    future_predictions = []
    last_60_days = scaled_data[-60:]
    for _ in range(days):
        x_input = np.array([last_60_days])
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
        pred_price = model.predict(x_input)
        future_predictions.append(pred_price[0][0])
        last_60_days = np.append(last_60_days[1:], pred_price, axis=0)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

def check_profit_criteria(ticker, interval, profit_target, period):
    df = fetch_stock_data(ticker, interval)
    df = predict_stock_prices(df)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    logging.info(f"Checking profit criteria for {ticker} with target {profit_target} over period {period}")
    
    if period == 'day':
        period_days = 1
    elif period == 'week':
        period_days = 7
    elif period == 'month':
        period_days = 30
    elif period == 'quarter':
        period_days = 90
    
    for i in range(len(df) - period_days):
        buy_price = df['Close'].iloc[i]
        sell_price = df['Close'].iloc[i + period_days]
        total_profit = sell_price - buy_price
        
        logging.info(f"Buy price: {buy_price}, Sell price: {sell_price}, Total profit: {total_profit}")
        
        if total_profit >= profit_target:
            buy_date = df.index[i]
            sell_date = df.index[i + period_days]
            logging.info(f"Found matching criteria for {ticker} with profit {total_profit}")
            return total_profit, buy_date, sell_date
    
    logging.info(f"No matching criteria found for {ticker}")
    return None, None, None

@app.route('/get_stock_data', methods=['GET'])
@login_required
def get_stock_data():
    try:
        ticker = request.args.get('ticker')
        interval = request.args.get('interval', '1d')
        stock_type = request.args.get('stock_type', 'profit')
        df = fetch_stock_data(ticker, interval)
        df = predict_stock_prices(df)
        if stock_type == 'profit':
            df['Profit'] = df['Close'] - df['Open']
            actual_data = df[['Date', 'Open', 'Close', 'High', 'Low', 'Profit']].to_dict('records')
        else:
            df['Interest'] = df['Close'] * 0.01
            actual_data = df[['Date', 'Open', 'Close', 'High', 'Low', 'Interest']].to_dict('records')
        predicted_data = df[['Date', 'Predicted']].to_dict('records')
        return jsonify({'actual': actual_data, 'predicted': predicted_data})
    except Exception as e:
        logging.error(f"Error in /get_stock_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        symbol = request.form['symbol']
        df = fetch_stock_data(symbol, '1d')
        x_train, y_train, scaler, training_data_len, scaled_data, full_df = preprocess_data(df)
        model = build_lstm_model(x_train, y_train)
        future_predictions = predict_future_prices(model, scaler, scaled_data, days=730)
        
        future_dates = pd.date_range(start=pd.to_datetime(full_df['Date'].iloc[-1]) + pd.Timedelta(days=1), periods=730, freq='D')
        future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predictions'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full_df['Date'], y=full_df['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predictions'], mode='lines', name='Predictions'))
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify(graphJSON=graphJSON, symbol=symbol)
    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_profit_stocks', methods=['GET'])
@login_required
def get_profit_stocks():
    try:
        profit_target = float(request.args.get('profit_target'))
        period = request.args.get('period', 'day')
        interval = request.args.get('interval', '1d')
        stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        matching_stocks = []
        for ticker in stock_list:
            total_profit, buy_date, sell_date = check_profit_criteria(ticker, interval, profit_target, period)
            if buy_date and sell_date:
                matching_stocks.append({
                    'ticker': ticker,
                    'total_profit': total_profit,
                    'buy_date': buy_date.strftime('%Y-%m-%d'),
                    'sell_date': sell_date.strftime('%Y-%m-%d')
                })
                logging.info(f"Appending matching stock: {ticker}")
        if not matching_stocks:
            logging.info("No stocks matched the criteria.")
        return jsonify({'matching_stocks': matching_stocks})
    except ValueError as ve:
        logging.error(f"ValueError in /get_profit_stocks: {str(ve)}")
        return jsonify({'error': 'Invalid input value.'}), 500
    except Exception as e:
        logging.error(f"Error in /get_profit_stocks: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different username.', 'danger')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
