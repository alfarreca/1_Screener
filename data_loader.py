import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    return None

@st.cache_data
def fetch_yfinance_data(symbols):
    if not symbols:
        return pd.DataFrame()
    
    # Get current date and date from 1 year ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        data = yf.download(
            tickers=list(symbols),
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            group_by='ticker',
            progress=False
        )
        
        # Process the data to get relevant metrics
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                result = {
                    'Symbol': symbol,
                    'Current Price': info.get('currentPrice', info.get('regularMarketPrice', None)),
                    '52 Week High': info.get('fiftyTwoWeekHigh', None),
                    '52 Week Low': info.get('fiftyTwoWeekLow', None),
                    'PE Ratio': info.get('trailingPE', None),
                    'Market Cap': info.get('marketCap', None),
                    'Dividend Yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
                    'Beta': info.get('beta', None),
                    'Volume': info.get('volume', None),
                    'Avg Volume': info.get('averageVolume', None),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A')
                }
                results.append(result)
            except:
                st.warning(f"Could not fetch data for {symbol}")
                continue
        
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()
