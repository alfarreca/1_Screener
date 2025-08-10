import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# App configuration
st.set_page_config(page_title="Financial Stock Screener", layout="wide")

# Cache data to avoid reloading
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        # Convert filter columns to strings to handle mixed types
        for col in ['Sector', 'Industry Group', 'Industry']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('N/A')
        return df
    return None

@st.cache_data
def fetch_yfinance_data(symbols, start_date, end_date):
    if not symbols:
        return pd.DataFrame()
    
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

def create_technical_chart(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
            
        # Calculate indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          subplot_titles=(f"{symbol} Price", "Volume", "RSI (14)"),
                          row_heights=[0.6, 0.2, 0.2])
        
        # Price plot
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'], 
            name='Price',
            line=dict(color='royalblue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['SMA_20'], 
            name='20-day SMA',
            line=dict(color='orange', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['SMA_50'], 
            name='50-day SMA',
            line=dict(color='green', width=1.5)
        ), row=1, col=1)
        
        # Volume plot
        fig.add_trace(go.Bar(
            x=data.index, 
            y=data['Volume'], 
            name='Volume',
            marker_color='rgba(100, 149, 237, 0.6)'
        ), row=2, col=1)
        
        # RSI plot
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=rsi, 
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        # Add RSI reference lines
        fig.add_hline(y=70, row=3, col=1, line_dash="dash", line_color="red")
        fig.add_hline(y=30, row=3, col=1, line_dash="dash", line_color="green")
        
        fig.update_layout(
            height=800, 
            showlegend=True, 
            hovermode='x unified',
            template='plotly_white'
        )
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

# Main app
def main():
    st.title("Financial Stock Screener")
    st.write("Upload an Excel file with stock symbols and use filters to screen stocks.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success("File uploaded successfully!")
            
            # Display raw data option
            if st.checkbox("Show raw data"):
                st.dataframe(df)
            
            # Check required columns
            required_columns = {'Symbol', 'Sector', 'Industry Group', 'Industry'}
            if not required_columns.issubset(df.columns):
                st.error(f"Missing required columns. Your file needs these columns: {required_columns}")
                return
            
            # Create filters
            st.sidebar.header("Filters")
            
            # Date range selector
            start_date, end_date = st.sidebar.date_input(
                "Date range",
                value=[datetime.now() - timedelta(days=365), datetime.now()],
                max_value=datetime.now()
            )
            
            # Multi-select filters with string conversion
            selected_sectors = st.sidebar.multiselect(
                'Select Sectors',
                options=sorted(df['Sector'].astype(str).unique()),
                default=None
            )
            
            # Industry Group filter
            if selected_sectors:
                industry_groups = st.sidebar.multiselect(
                    'Select Industry Groups',
                    options=sorted(df[df['Sector'].isin(selected_sectors)]['Industry Group'].astype(str).unique()),
                    default=None
                )
            else:
                industry_groups = st.sidebar.multiselect(
                    'Select Industry Groups',
                    options=sorted(df['Industry Group'].astype(str).unique()),
                    default=None
                )
            
            # Industry filter
            if selected_sectors and industry_groups:
                industries = st.sidebar.multiselect(
                    'Select Industries',
                    options=sorted(df[(df['Sector'].isin(selected_sectors)) & 
                                   (df['Industry Group'].isin(industry_groups))]['Industry'].astype(str).unique()),
                    default=None
                )
            elif selected_sectors:
                industries = st.sidebar.multiselect(
                    'Select Industries',
                    options=sorted(df[df['Sector'].isin(selected_sectors)]['Industry'].astype(str).unique()),
                    default=None
                )
            elif industry_groups:
                industries = st.sidebar.multiselect(
                    'Select Industries',
                    options=sorted(df[df['Industry Group'].isin(industry_groups)]['Industry'].astype(str).unique()),
                    default=None
                )
            else:
                industries = st.sidebar.multiselect(
                    'Select Industries',
                    options=sorted(df['Industry'].astype(str).unique()),
                    default=None
                )
            
            # Apply filters
            filtered_df = df.copy()
            if selected_sectors:
                filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
            if industry_groups:
                filtered_df = filtered_df[filtered_df['Industry Group'].isin(industry_groups)]
            if industries:
                filtered_df = filtered_df[filtered_df['Industry'].isin(industries)]
            
            st.subheader("Filtered Stocks")
            st.write(f"Found {len(filtered_df)} stocks matching your criteria")
            
            if not filtered_df.empty:
                st.dataframe(filtered_df[['Symbol', 'Name', 'Sector', 'Industry Group', 'Industry']])
                
                # Button to fetch data
                if st.button("Fetch Financial Data from Yahoo Finance"):
                    with st.spinner("Fetching data from Yahoo Finance. This may take a while..."):
                        symbols = filtered_df['Symbol'].tolist()
                        financial_data = fetch_yfinance_data(symbols, start_date, end_date)
                        
                        if not financial_data.empty:
                            st.success("Data fetched successfully!")
                            
                            # Merge with original data
                            result_df = pd.merge(
                                filtered_df,
                                financial_data,
                                on='Symbol',
                                how='left'
                            )
                            
                            # Display results in tabs
                            tab1, tab2 = st.tabs(["Financial Data", "Technical Analysis"])
                            
                            with tab1:
                                st.dataframe(result_df)
                                
                                # Download button
                                csv = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name='stock_screener_results.csv',
                                    mime='text/csv'
                                )
                            
                            with tab2:
                                selected_stock = st.selectbox(
                                    "Select stock for technical analysis", 
                                    result_df['Symbol'],
                                    key='tech_analysis_select'
                                )
                                chart = create_technical_chart(selected_stock, start_date, end_date)
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                                else:
                                    st.warning("Could not generate technical analysis for this stock")
                        else:
                            st.warning("No financial data was fetched. Please check your symbols.")
            else:
                st.warning("No stocks match your filter criteria.")

if __name__ == "__main__":
    main()
