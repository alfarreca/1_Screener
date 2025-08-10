import streamlit as st
import pandas as pd
from data_loader import load_data, fetch_yfinance_data
from analysis import apply_filters, get_filter_options, validate_data

def show_main_app():
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
            
            # Validate data
            if not validate_data(df):
                return
            
            # Initialize session state for filters if not exists
            if 'selected_sector' not in st.session_state:
                st.session_state.selected_sector = 'All'
            if 'selected_industry_group' not in st.session_state:
                st.session_state.selected_industry_group = 'All'
            if 'selected_industry' not in st.session_state:
                st.session_state.selected_industry = 'All'
            
            # Create filters in sidebar
            st.sidebar.header("Filters")
            
            # Get filter options
            sectors, industry_groups, industries = get_filter_options(
                df, 
                st.session_state.selected_sector, 
                st.session_state.selected_industry_group
            )
            
            # Update filter widgets
            st.session_state.selected_sector = st.sidebar.selectbox('Select Sector', sectors)
            st.session_state.selected_industry_group = st.sidebar.selectbox('Select Industry Group', industry_groups)
            st.session_state.selected_industry = st.sidebar.selectbox('Select Industry', industries)
            
            # Apply filters
            filtered_df = apply_filters(
                df,
                st.session_state.selected_sector,
                st.session_state.selected_industry_group,
                st.session_state.selected_industry
            )
            
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
