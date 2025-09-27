import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import ta
import feedparser
import urllib.parse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import math
import time

st.set_page_config(page_title="IBEX 35 Analyzer", layout="wide")
st.title("📊 Análisis de Empresas - IBEX 35")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Opciones")
tickers_input = st.sidebar.text_area(
    "Introduce tickers separados por comas",
    "SAN.MC, BBVA.MC, TEF.MC, ITX.MC"
)
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

PERIOD = "3mo"
INTERVAL = "1d"
NEWS_PER_SOURCE = 5
TECH_THRESHOLD = 2
FUND_THRESHOLD = 2
NEWS_THRESHOLD = 0.05
PE_MAX = 35.0
DTE_MAX = 1.5
ROE_MIN = 0.05
PM_MIN = 0.03
RSI_MAX_FOR_BUY = 50.0

EMPRESAS = {
    "ANA.MC": "Acciona", "ACX.MC": "Acerinox", "ACS.MC": "ACS",
    "AENA.MC": "Aena", "ALM.MC": "Almirall", "AMS.MC": "Amadeus",
    "SAB.MC": "Banco Sabadell", "SAN.MC": "Banco Santander",
    "BBVA.MC": "BBVA", "CABK.MC": "CaixaBank", "CLNX.MC": "Cellnex",
    "CIE.MC": "CIE Automotive", "COL.MC": "Colonial", "ENG.MC": "Enagás",
    "ELE.MC": "Endesa", "FER.MC": "Ferrovial", "GRF.MC": "Grifols",
    "IBE.MC": "Iberdrola", "ITX.MC": "Inditex", "IDR.MC": "Indra",
    "MAP.MC": "Mapfre", "MEL.MC": "Meliá Hotels", "NTGY.MC": "Naturgy",
    "PHM.MC": "PharmaMar", "RED.MC": "Redeia", "REP.MC": "Repsol",
    "ROVI.MC": "Rovi", "SLR.MC": "Solaria", "TEF.MC": "Telefónica",
    "UNI.MC": "Unicaja Banco", "VIS.MC": "Viscofan", "LOG.MC": "Logista"
}

analyzer = SentimentIntensityAnalyzer()

# -------------------------
# Funciones
# -------------------------
def obtener_feed_google_news(query):
    query_enc = urllib.parse.quote(query)
    url_es = f"https://news.google.com/rss/search?q={query_enc}&hl=es&gl=ES&ceid=ES:ES"
    url_en = f"https://news.google.com/rss/search?q={query_enc}&hl=en&gl=US&ceid=US:en"
    return [url_es, url_en]

def evaluar_noticias(ticker, n=NEWS_PER_SOURCE):
    nombre = EMPRESAS.get(ticker)
    if not nombre:
        return 0
    score_total = 0
    feeds = obtener_feed_google_news(nombre)
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:n]:
            vs = analyzer.polarity_scores(entry.title)
            score_total += vs['compound']
    total_items = len(feeds) * n
    return score_total / total_items if total_items > 0 else 0

def es_bullish_engulfing(df):
    if len(df) < 2: return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    return prev['Close'] < prev['Open'] and last['Close'] > last['Open'] and last['Close'] > prev['Open'] and last['Open'] < prev['Close']

def evaluar_fundamentales(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        return {"score": 0, "details": {}, "raw": {}}
    score = 0
    details = {}
    pe = info.get('trailingPE') or info.get('forwardPE')
    score += 1 if pe and 0 < pe < PE_MAX else 0; details['PE_ok']= bool(score)
    dte = info.get('debtToEquity') or info.get('totalDebt')
    dte_val = float(dte) if dte else None
    score += 1 if dte_val and dte_val < DTE_MAX else 0; details['D/E_ok']= bool(dte_val and dte_val < DTE_MAX)
    roe = info.get('returnOnEquity') or info.get('returnOnInvestment')
    roe_val = float(roe) if roe else None
    score += 1 if roe_val and roe_val > ROE_MIN else 0; details['ROE_ok']= bool(roe_val and roe_val > ROE_MIN)
    pm = info.get('profitMargins')
    pm_val = float(pm) if pm else None
    score += 1 if pm_val and pm_val > PM_MIN else 0; details['PM_ok']= bool(pm_val and pm_val > PM_MIN)
    return {"score": score, "details": details, "raw": info}

def evaluar_tecnica(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
        if df.empty or 'Close' not in df.columns: return None
        close = df['Close'].astype(float)
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd_obj = ta.trend.MACD(close)
        macd, macd_signal = macd_obj.macd(), macd_obj.macd_signal()
        bull = es_bullish_engulfing(df)
        tech_score = sum([
            rsi.iloc[-1]<RSI_MAX_FOR_BUY,
            ema20.iloc[-1]>=ema50.iloc[-1],
            macd.iloc[-1]>macd_signal.iloc[-1],
            bull
        ])
        return {'RSI':rsi.iloc[-1],'EMA20':ema20.iloc[-1],'EMA50':ema50.iloc[-1],'MACD':macd.iloc[-1],'MACD_signal':macd_signal.iloc[-1],'Bullish':bull,'tech_score':tech_score}
    except:
        return None

def calcular_metricas(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d")['Close'].dropna()
        rets = data.pct_change().dropna()
        sharpe = (rets.mean()/rets.std())*np.sqrt(252)
        max_dd = ((data/data.cummax())-1).min()
        return round(sharpe,2), round(max_dd,2)
    except:
        return None, None

def optimizar_cartera(tickers):
    try:
        precios = yf.download(tickers, period="1y", interval="1d")['Close'].dropna(axis=1)
        mu = expected_returns.mean_historical_return(precios)
        S = risk_models.sample_cov(precios)
        ef = EfficientFrontier(mu,S)
        ef.max_sharpe()
        return ef.clean_weights()
    except:
        return {}

# -------------------------
# Main Streamlit
# -------------------------
st.header("📈 Resultados por Empresa")
resultados = []

for t in tickers:
    tech = evaluar_tecnica(t)
    fund = evaluar_fundamentales(t)
    news_score = evaluar_noticias(t)
    tech_score = tech['tech_score'] if tech else 0
    fund_score = fund['score'] if fund else 0
    etiqueta=None
    if tech_score>=TECH_THRESHOLD and fund_score>=FUND_THRESHOLD and news_score>=NEWS_THRESHOLD:
        etiqueta="STRONG BUY"
    elif tech_score>=TECH_THRESHOLD or fund_score>=FUND_THRESHOLD or news_score>0:
        etiqueta="BUY"

    if etiqueta:
        sharpe, dd = calcular_metricas(t)
        resultados.append({
            'Ticker': t,
            'Empresa': EMPRESAS.get(t),
            'Etiqueta': etiqueta,
            'TechScore': tech_score,
            'FundScore': fund_score,
            'NewsScore': round(news_score,3),
            'RSI': round(tech['RSI'],2) if tech else math.nan,
            'Sharpe': sharpe,
            'Drawdown': dd
        })

if resultados:
    df = pd.DataFrame(resultados).sort_values(by=['TechScore','FundScore','NewsScore'], ascending=False)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", csv, "ranking_empresas.csv", "text/csv")
else:
    st.warning("No se encontraron resultados para los tickers introducidos.")
