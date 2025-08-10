import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(
    page_title="Financial Stock Screener Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache configuration
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(uploaded_file):
    """Load and preprocess Excel file"""
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            # Convert filter columns to strings and handle missing values
            for col in ['Sector', 'Industry Group', 'Industry']:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('N/A')
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Failed to load the Excel file. Please check the file format.")
        return None

@st.cache_data(ttl=1800, show_spinner="Fetching market data...")
def fetch_yfinance_data(symbols, start_date, end_date):
    """Fetch data from Yahoo Finance with error handling"""
    if not symbols:
        return pd.DataFrame()
    
    try:
        data = yf.download(
            tickers=list(symbols),
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            group_by='ticker',
            progress=False,
            timeout=10  # Add timeout
        )
        
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
                    'Market Cap': f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else None,
                    'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else None,
                    'Beta': info.get('beta', None),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A')
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Yahoo Finance error: {str(e)}")
        st.error("Failed to fetch market data. Please try again later.")
        return pd.DataFrame()

def create_technical_chart(symbol, start_date, end_date):
    """Generate interactive technical analysis chart"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
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
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": False}], [{}], [{}]]
        )
        
        # Price plot
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color='#3366CC', width=2),
                hovertemplate='%{y:.2f}<extra>Price</extra>'
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='20-day SMA',
                line=dict(color='#FF9900', width=1.5),
                hovertemplate='%{y:.2f}<extra>20-day SMA</extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='50-day SMA',
                line=dict(color='#109618', width=1.5),
                hovertemplate='%{y:.2f}<extra>50-day SMA</extra>'
            ),
            row=1, col=1
        )
        
        # Volume plot
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='#7F7F7F',
                opacity=0.6,
                hovertemplate='%{y:,}<extra>Volume</extra>'
            ),
            row=2, col=1
        )
        
        # RSI plot
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                name='RSI (14)',
                line=dict(color='#990099', width=1.5),
                hovertemplate='%{y:.2f}<extra>RSI</extra>'
            ),
            row=3, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(
            y=70, row=3, col=1,
            line=dict(color='#DC3912', width=1, dash='dot'),
            annotation_text=' Overbought', annotation_position='top right'
        )
        fig.add_hline(
            y=30, row=3, col=1,
            line=dict(color='#109618', width=1, dash='dot'),
            annotation_text=' Oversold', annotation_position='bottom right'
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            margin=dict(t=40, b=40, l=40, r=40),
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='rgba(240,240,240,0.8)',
            paper_bgcolor='rgba(240,240,240,0.1)'
        )
        
        # Format axes
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='RSI', range=[0,100], row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    except Exception as e:
        logger.error(f"Chart error for {symbol}: {str(e)}")
        return None

def main():
    try:
        st.title("📊 Financial Stock Screener Pro")
        st.caption("Upload an Excel file with stock symbols to analyze market data")
        
        # File upload section
        with st.expander("📁 Upload Excel File", expanded=True):
            uploaded_file = st.file_uploader(
                "Choose file",
                type=['xlsx', 'xls'],
                help="File must contain: Symbol, Sector, Industry Group, Industry columns"
            )
        
        if uploaded_file:
            df = load_data(uploaded_file)
            
            if df is not None:
                # Check required columns
                required_columns = {'Symbol', 'Sector', 'Industry Group', 'Industry'}
                if not required_columns.issubset(df.columns):
                    st.error(f"Missing required columns: {required_columns - set(df.columns)}")
                    return
                
                # Filter section
                with st.sidebar:
                    st.header("🔍 Filters")
                    
                    # Date range selector
                    start_date, end_date = st.date_input(
                        "Date range",
                        value=[datetime.now() - timedelta(days=365), datetime.now()],
                        max_value=datetime.now()
                    )
                    
                    # Multi-select filters
                    selected_sectors = st.multiselect(
                        'Select Sectors',
                        options=sorted(df['Sector'].unique()),
                        placeholder="All sectors"
                    )
                    
                    # Industry Group filter
                    industry_groups = st.multiselect(
                        'Select Industry Groups',
                        options=sorted(df[df['Sector'].isin(selected_sectors)]['Industry Group'].unique() 
                        if selected_sectors else sorted(df['Industry Group'].unique()),
                        placeholder="All industry groups"
                    )
                    
                    # Industry filter
                    industries = st.multiselect(
                        'Select Industries',
                        options=sorted(df[
                            (df['Sector'].isin(selected_sectors) & 
                            (df['Industry Group'].isin(industry_groups))
                        ]['Industry'].unique() if (selected_sectors and industry_groups) else
                        sorted(df[df['Sector'].isin(selected_sectors)]['Industry'].unique() 
                        if selected_sectors else
                        sorted(df[df['Industry Group'].isin(industry_groups)]['Industry'].unique() 
                        if industry_groups else sorted(df['Industry'].unique()),
                        placeholder="All industries"
                    )
                
                # Apply filters
                filtered_df = df.copy()
                if selected_sectors:
                    filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
                if industry_groups:
                    filtered_df = filtered_df[filtered_df['Industry Group'].isin(industry_groups)]
                if industries:
                    filtered_df = filtered_df[filtered_df['Industry'].isin(industries)]
                
                # Display filtered stocks
                with st.expander(f"📋 Filtered Stocks ({len(filtered_df)})", expanded=True):
                    st.dataframe(
                        filtered_df[['Symbol', 'Name', 'Sector', 'Industry Group', 'Industry']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Fetch and display market data
                if st.button("🚀 Fetch Market Data", type="primary"):
                    with st.spinner("Loading market data..."):
                        symbols = filtered_df['Symbol'].tolist()
                        financial_data = fetch_yfinance_data(symbols, start_date, end_date)
                        
                        if not financial_data.empty:
                            # Merge with original data
                            result_df = pd.merge(
                                filtered_df,
                                financial_data,
                                on='Symbol',
                                how='left'
                            )
                            
                            # Display results in tabs
                            tab1, tab2 = st.tabs(["💵 Financial Data", "📈 Technical Analysis"])
                            
                            with tab1:
                                st.dataframe(
                                    result_df.style.format({
                                        'Current Price': '{:.2f}',
                                        '52 Week High': '{:.2f}',
                                        '52 Week Low': '{:.2f}',
                                        'PE Ratio': '{:.2f}',
                                        'Beta': '{:.2f}'
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Download button
                                csv = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=csv,
                                    file_name='stock_screener_results.csv',
                                    mime='text/csv',
                                    type="secondary"
                                )
                            
                            with tab2:
                                selected_stock = st.selectbox(
                                    "Select stock for technical analysis",
                                    result_df['Symbol'],
                                    index=0,
                                    key="tech_analysis_select"
                                )
                                
                                chart = create_technical_chart(selected_stock, start_date, end_date)
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                                else:
                                    st.warning("Technical analysis not available for this stock")
                        else:
                            st.warning("No market data found for the selected stocks")
    
    except Exception as e:
        logger.error(f"App crashed: {str(e)}")
        st.error("The application encountered an unexpected error. Please try again later.")
        st.stop()

if __name__ == "__main__":
    main()
