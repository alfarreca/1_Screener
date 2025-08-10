import pandas as pd
import streamlit as st

def apply_filters(df, selected_sector, selected_industry_group, selected_industry):
    filtered_df = df.copy()
    if selected_sector != 'All':
        filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
    if selected_industry_group != 'All':
        filtered_df = filtered_df[filtered_df['Industry Group'] == selected_industry_group]
    if selected_industry != 'All':
        filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]
    return filtered_df

def get_filter_options(df, selected_sector, selected_industry_group):
    # Sector filter options
    sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
    
    # Industry Group filter options
    if selected_sector != 'All':
        industry_groups = ['All'] + sorted(df[df['Sector'] == selected_sector]['Industry Group'].dropna().unique().tolist())
    else:
        industry_groups = ['All'] + sorted(df['Industry Group'].dropna().unique().tolist())
    
    # Industry filter options
    if selected_sector != 'All' and selected_industry_group != 'All':
        industries = ['All'] + sorted(df[(df['Sector'] == selected_sector) & 
                                      (df['Industry Group'] == selected_industry_group)]['Industry'].dropna().unique().tolist())
    elif selected_sector != 'All':
        industries = ['All'] + sorted(df[df['Sector'] == selected_sector]['Industry'].dropna().unique().tolist())
    elif selected_industry_group != 'All':
        industries = ['All'] + sorted(df[df['Industry Group'] == selected_industry_group]['Industry'].dropna().unique().tolist())
    else:
        industries = ['All'] + sorted(df['Industry'].dropna().unique().tolist())
    
    return sectors, industry_groups, industries

def validate_data(df):
    required_columns = {'Symbol', 'Sector', 'Industry Group', 'Industry'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing required columns. Your file needs these columns: {required_columns}")
        return False
    return True
