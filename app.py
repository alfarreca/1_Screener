import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# App configuration
st.set_page_config(page_title="Financial Stock Screener", layout="wide")

# Cache data to avoid reloading
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

# Main app
def main():
    st.title("Financial Stock Screener")
    st.write("Upload an Excel file with stock symbols and use filters to screen stocks. Data will only be fetched from Yahoo Finance after applying filters.")
    
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
            
            # Sector filter
            sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
            selected_sector = st.sidebar.selectbox('Select Sector', sectors)
            
            # Industry Group filter (based on selected sector)
            if selected_sector != 'All':
                industry_groups = ['All'] + sorted(df[df['Sector'] == selected_sector]['Industry Group'].dropna().unique().tolist())
            else:
                industry_groups = ['All'] + sorted(df['Industry Group'].dropna().unique().tolist())
            selected_industry_group = st.sidebar.selectbox('Select Industry Group', industry_groups)
            
            # Industry filter (based on selected sector and industry group)
            if selected_sector != 'All' and selected_industry_group != 'All':
                industries = ['All'] + sorted(df[(df['Sector'] == selected_sector) & 
                                              (df['Industry Group'] == selected_industry_group)]['Industry'].dropna().unique().tolist())
            elif selected_sector != 'All':
                industries = ['All'] + sorted(df[df['Sector'] == selected_sector]['Industry'].dropna().unique().tolist())
            elif selected_industry_group != 'All':
                industries = ['All'] + sorted(df[df['Industry Group'] == selected_industry_group]['Industry'].dropna().unique().tolist())
            else:
                industries = ['All'] + sorted(df['Industry'].dropna().unique().tolist())
            selected_industry = st.sidebar.selectbox('Select Industry', industries)
            
            # Apply filters
            filtered_df = df.copy()
            if selected_sector != 'All':
                filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
            if selected_industry_group != 'All':
                filtered_df = filtered_df[filtered_df['Industry Group'] == selected_industry_group]
            if selected_industry != 'All':
                filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]
            
            st.subheader("Filtered Stocks")
            st.write(f"Found {len(filtered_df)} stocks matching your criteria")
            
            if not filtered_df.empty:
                st.dataframe(filtered_df[['Symbol', 'Name', 'Sector', 'Industry Group', 'Industry']])
                
                # Button to fetch data
                if st.button("Fetch Financial Data from Yahoo Finance"):
                    with st.spinner("Fetching data from Yahoo Finance. This may take a while..."):
                        symbols = filtered_df['Symbol'].tolist()
                        financial_data = fetch_yfinance_data(symbols)
                        
                        if not financial_data.empty:
                            st.success("Data fetched successfully!")
                            
                            # Merge with original data
                            result_df = pd.merge(
                                filtered_df,
                                financial_data,
                                on='Symbol',
                                how='left'
                            )
                            
                            # Display results
                            st.dataframe(result_df)
                            
                            # Download button
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name='stock_screener_results.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No financial data was fetched. Please check your symbols.")
            else:
                st.warning("No stocks match your filter criteria.")

if __name__ == "__main__":
    main()
