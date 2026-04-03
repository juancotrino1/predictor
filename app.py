"""
app.py  —  BTC 15-min predictor deployed on Render
===================================================
• Background thread runs the predictor on :02/:17/:32/:47 (ET)
• Flask serves:
    GET /          → dashboard (HTML)
    GET /stream    → Server-Sent Events live log
    GET /api/predictions → JSON list of all stored predictions
"""

import os
import math
import time
import queue
import threading
import warnings
import datetime
import traceback
import json

import numpy as np
import pandas as pd
import requests
import pytz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from flask import Flask, Response, jsonify, render_template

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Constants ──────────────────────────────────────────────────────────────────
ET          = pytz.timezone("America/New_York")
RUN_MINUTES = [2, 17, 32, 47]

FEATURES  = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA7", "SMA15", "SMA30",
    "EMA7", "EMA15", "EMA30",
    "RSI", "Upper_Band", "Lower_Band",
]
CLOSE_IDX = FEATURES.index("Close")
SEQ_LEN   = 20

ARIMA_EXOG = [
    "Daily_Return", "Volume", "MACD", "Signal", "RSI",
    "Middle_Band", "Upper_Band", "Lower_Band", "SMA7", "EMA7",
]

# In-memory store  (Render's free tier has no persistent disk)
predictions: list[dict] = []

# SSE broadcast queue pool
_sse_queues: list[queue.Queue] = []
_sse_lock = threading.Lock()


# ── Logging helper (prints + broadcasts to SSE clients) ───────────────────────
def log(msg: str):
    ts  = datetime.datetime.now(pytz.utc).astimezone(ET).strftime("%H:%M:%S ET")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    data = json.dumps({"msg": line})
    with _sse_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


# ── Time helpers ───────────────────────────────────────────────────────────────
def now_et():
    return datetime.datetime.now(pytz.utc).astimezone(ET)

def next_interval(now):
    boundary_min = math.ceil(now.minute / 15) * 15
    extra_hours  = boundary_min // 60
    boundary_min %= 60
    start = (
        now.replace(minute=0, second=0, microsecond=0)
        + datetime.timedelta(hours=extra_hours, minutes=boundary_min)
    )
    return start, start + datetime.timedelta(minutes=15)

def seconds_until_next_run():
    now = now_et()
    for rm in RUN_MINUTES:
        if rm > now.minute or (rm == now.minute and now.second < 5):
            target = now.replace(minute=rm, second=0, microsecond=0)
            return max(0.0, (target - now).total_seconds())
    target = (
        now.replace(minute=RUN_MINUTES[0], second=0, microsecond=0)
        + datetime.timedelta(hours=1)
    )
    return max(0.0, (target - now).total_seconds())


# ── Data fetching ──────────────────────────────────────────────────────────────
def fetch_and_prepare() -> pd.DataFrame:
    BINANCE_URL = "https://api.binance.com/api/v3/klines"
    SYMBOL, INTERVAL, LIMIT = "BTCUSDT", "15m", 1000

    all_rows, end_time = [], None
    for _ in range(3):
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(BINANCE_URL, params=params, timeout=15)
        resp.raise_for_status()
        candles = resp.json()
        if not candles:
            break
        all_rows = candles + all_rows
        end_time = candles[0][0] - 1

    records = [{
        "Open":      float(c[1]), "High":  float(c[2]),
        "Low":       float(c[3]), "Close": float(c[4]),
        "Volume":    float(c[5]),
        "timestamp": pd.Timestamp(c[0], unit="ms", tz="UTC"),
    } for c in all_rows]

    df = pd.DataFrame(records).set_index("timestamp")
    df.index = df.index.tz_convert(ET)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for w in [7, 15, 30]:
        df[f"SMA{w}"] = df["Close"].rolling(w).mean().fillna(df["Close"])
        df[f"EMA{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    rs    = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    df["Middle_Band"] = df["Close"].rolling(15).mean()
    std15 = df["Close"].rolling(15).std()
    df["Upper_Band"]  = (df["Middle_Band"] + 2 * std15).fillna(df["Close"])
    df["Lower_Band"]  = (df["Middle_Band"] - 2 * std15).fillna(df["Close"])

    ema12, ema26   = df["Close"].ewm(span=12, adjust=False).mean(), df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]     = ema12 - ema26
    df["Signal"]   = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Daily_Return"] = df["Close"].pct_change() * 100

    df.dropna(inplace=True)
    return df


# ── Sequence helpers ───────────────────────────────────────────────────────────
def build_sequences(values, seq_len):
    return np.array([values[i:i+seq_len] for i in range(len(values) - seq_len + 1)])

def split_sequences(seqs):
    n = len(seqs)
    t = int(n * 0.8)
    v = int((n - t) * 0.5)
    return seqs[:t], seqs[t:t+v], seqs[t+v:]


# ── ARIMAX ─────────────────────────────────────────────────────────────────────
def predict_arimax(df):
    df_a = df.copy()
    df_a.index = df_a.index.tz_localize(None)
    y = df_a["Close"]
    X = df_a[[c for c in ARIMA_EXOG if c in df_a.columns]]
    d_order   = 0 if adfuller(y)[1] <= 0.05 else 1
    train_size = int(len(y) * 0.8)
    fit = SARIMAX(
        y.iloc[:train_size], exog=X.iloc[:train_size],
        order=(1, d_order, 1),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=1000)
    pred_price = float(fit.forecast(steps=1, exog=X.iloc[[-1]]).iloc[0])
    return round(pred_price, 2), "UP" if pred_price > float(y.iloc[-1]) else "DOWN"


# ── TensorFlow LSTM ────────────────────────────────────────────────────────────
def predict_tf_lstm(df):
    df_s = df[FEATURES].copy().dropna()
    scalers = {}
    for feat in FEATURES:
        sc = MinMaxScaler((-1, 1))
        df_s[feat] = sc.fit_transform(df_s[[feat]])
        scalers[feat] = sc

    seqs = build_sequences(df_s.values, SEQ_LEN)
    tr, va, te = split_sequences(seqs)

    x_tr = np.clip(tr[:, :-1, :] + np.random.normal(0, 0.01, tr[:, :-1, :].shape), -1, 1)
    y_tr = tr[:, -1, CLOSE_IDX]
    x_va, y_va = va[:, :-1, :], va[:, -1, CLOSE_IDX]
    x_te = te[:, :-1, :]

    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(x_tr.shape[1], x_tr.shape[2])),
        Dropout(0.5), Flatten(), Dense(5), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        x_tr, y_tr, epochs=50, batch_size=32,
        validation_data=(x_va, y_va), verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=15, min_lr=1e-5),
        ],
    )
    last  = x_te[-1:] if len(x_te) > 0 else x_va[-1:]
    sc_c  = scalers["Close"]
    pred  = float(sc_c.inverse_transform([[float(model.predict(last, verbose=0)[0, 0])]])[0, 0])
    last_c = float(sc_c.inverse_transform([[seqs[-1, -1, CLOSE_IDX]]])[0, 0])
    return round(pred, 2), "UP" if pred > last_c else "DOWN"


# ── PyTorch LSTM ───────────────────────────────────────────────────────────────
class _LSTMNet(nn.Module):
    def __init__(self, input_size, hidden=256, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(h[-1]))

def predict_pytorch_lstm(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_s = df[FEATURES].copy().dropna()
    scalers = {}
    for feat in FEATURES:
        sc = MinMaxScaler((-1, 1))
        df_s[feat] = sc.fit_transform(df_s[[feat]])
        scalers[feat] = sc

    seqs = build_sequences(df_s.values, SEQ_LEN)
    tr, va, te = split_sequences(seqs)

    x_tr = np.clip(tr[:, :-1, :] + np.random.normal(0, 0.01, tr[:, :-1, :].shape), -1, 1)
    y_tr = tr[:, -1, CLOSE_IDX]
    x_va, y_va = va[:, :-1, :], va[:, -1, CLOSE_IDX]

    def loader(X, y, shuffle=False):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=16, shuffle=shuffle)

    net  = _LSTMNet(x_tr.shape[2]).to(device)
    opt  = optim.Adam(net.parameters())
    crit = nn.MSELoss()
    best_val, patience_cnt, best_state = float("inf"), 0, None

    for _ in range(50):
        net.train()
        for Xb, yb in loader(x_tr, y_tr, True):
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(); crit(net(Xb), yb.unsqueeze(-1)).backward(); opt.step()
        net.eval()
        vl = sum(crit(net(Xb.to(device)), yb.to(device).unsqueeze(-1)).item()
                 for Xb, yb in loader(x_va, y_va)) / len(loader(x_va, y_va))
        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= 5: break
            if patience_cnt >= 3:
                for pg in opt.param_groups: pg["lr"] = max(pg["lr"] * 0.5, 1e-5)

    net.load_state_dict(best_state); net.eval()
    x_te = te[:, :-1, :]
    last = x_te[-1:] if len(x_te) > 0 else x_va[-1:]
    with torch.no_grad():
        pred_s = net(torch.FloatTensor(last).to(device)).item()
    sc_c   = scalers["Close"]
    pred   = float(sc_c.inverse_transform([[pred_s]])[0, 0])
    last_c = float(sc_c.inverse_transform([[seqs[-1, -1, CLOSE_IDX]]])[0, 0])
    return round(pred, 2), "UP" if pred > last_c else "DOWN"


# ── Core prediction job ────────────────────────────────────────────────────────
def run_prediction():
    t = now_et()
    int_start, int_end = next_interval(t)
    ts_gen   = t.strftime("%Y-%m-%d %H:%M:%S %Z")
    ts_start = int_start.strftime("%Y-%m-%d %H:%M %Z")
    ts_end   = int_end.strftime("%Y-%m-%d %H:%M %Z")

    log(f"── Predicting {ts_start} → {ts_end} ──")
    try:
        log("Fetching data from Binance…")
        df = fetch_and_prepare()
        log(f"{len(df)} candles loaded. Last close: ${df['Close'].iloc[-1]:,.2f}")

        log("Training TF LSTM…")
        tf_price, tf_dir = predict_tf_lstm(df)
        log(f"TF LSTM      → {tf_dir}  ${tf_price:,.2f}")

        log("Training PyTorch LSTM…")
        pt_price, pt_dir = predict_pytorch_lstm(df)
        log(f"PyTorch LSTM → {pt_dir}  ${pt_price:,.2f}")

        log("Fitting ARIMAX…")
        ar_price, ar_dir = predict_arimax(df)
        log(f"ARIMAX       → {ar_dir}  ${ar_price:,.2f}")

        row = {
            "generated_at":     ts_gen,
            "interval_start":   ts_start,
            "interval_end":     ts_end,
            "tf_price":         tf_price,
            "tf_dir":           tf_dir,
            "pt_price":         pt_price,
            "pt_dir":           pt_dir,
            "ar_price":         ar_price,
            "ar_dir":           ar_dir,
            "last_close":       round(float(df["Close"].iloc[-1]), 2),
        }
        predictions.append(row)
        # Keep last 200 in memory
        if len(predictions) > 200:
            predictions.pop(0)

        log(f"Done. {len(predictions)} predictions stored.")

        # Broadcast the new row to SSE clients as a special event
        data = json.dumps({"new_prediction": row})
        with _sse_lock:
            for q in _sse_queues:
                try: q.put_nowait(data)
                except queue.Full: pass

    except Exception:
        log("ERROR: " + traceback.format_exc().splitlines()[-1])
        traceback.print_exc()


# ── Background scheduler thread ────────────────────────────────────────────────
def scheduler():
    log("Scheduler started. First run now…")
    run_prediction()
    while True:
        wait = seconds_until_next_run()
        t_next = (now_et() + datetime.timedelta(seconds=wait)).strftime("%H:%M:%S %Z")
        log(f"Sleeping {wait:.0f}s — next run at {t_next}")
        time.sleep(wait)
        run_prediction()


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predictions")
def api_predictions():
    return jsonify(list(reversed(predictions)))

@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — one client queue per connection."""
    def generate():
        q = queue.Queue(maxsize=50)
        with _sse_lock:
            _sse_queues.append(q)
        try:
            # Immediately send last 5 stored predictions as seed
            for row in predictions[-5:]:
                yield f"data: {json.dumps({'new_prediction': row})}\n\n"
            while True:
                try:
                    data = q.get(timeout=25)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"   # keep-alive
        finally:
            with _sse_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=scheduler, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
