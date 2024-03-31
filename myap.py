import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
# Function to find the start row where data begins based on the "Balance" column presence
def find_data_start_row(file_path):
    for skip_rows in range(50):
        df = pd.read_excel(file_path, header=skip_rows, nrows=1)
        if 'Closing Balance' in df.columns or any('balance' in str(col).lower() for col in df.columns):
            return skip_rows
    return None

# Function to load the bank statement into a DataFrame
def load_bank_statement(file_path):
    start_row = find_data_start_row(file_path)
    if start_row is not None:
        df = pd.read_excel(file_path, header=start_row)
        return df
    else:
        raise ValueError("Failed to automatically detect the start row of the data.")


def load_sample_data():
    # Here you would define your sample data
    # For the sake of example, I'm generating a simple DataFrame
    # Replace this with your actual sample data
    data = {
        'Date': pd.date_range(start='1/1/2022', periods=10, freq='M'),
        'Transaction ID': [f'TID{100+i}' for i in range(10)],
        'Details': ['Sample Data' for _ in range(10)],
        'Withdrawal Amount ($)': [100, 200, np.nan, 150, np.nan, 300, 250, np.nan, 400, 100],
        'Deposited Amount ($)': [np.nan, np.nan, 500, np.nan, 1000, np.nan, np.nan, 200, np.nan, np.nan]
    }
    sample_df = pd.DataFrame(data)
    return sample_df


# Streamlit application starts here
st.title('Visualize Your Bank Statement')
st.write("This app include the ability to take records from e-banking statement and provide useful financial summaries while showing the trends in spending. It does everything at the best by automating the transaction categorization.")
uploaded_file = st.file_uploader("Choose an xls format file of your Bank Statement", type=["xls", "xlsx"])

if uploaded_file is not None:
    df = load_bank_statement(uploaded_file)
    # Ensure 'Date' column is in datetime format for analysis
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert invalid or missing dates to NaT 
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date

    # Filter out rows where 'Date' is NaT
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= pd.to_datetime('1990-01-01')]
    df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
    

    if not df.empty:
        # Proceed with the analysis only if DataFrame is not empty after dropping NaT dates
        df['Date'] = pd.to_datetime(df['Date'])
        start_date = df['Date'].min().strftime("%B %d %Y")
        end_date = df['Date'].max().strftime("%B %d %Y")
        st.write(f"Statement Period: {start_date} to {end_date}")

        days = (df['Date'].max() - df['Date'].min()).days
        numeric_columns = df.select_dtypes(include='number').columns
        # Perform operations on selected numeric columns
        col1, col2= st.columns(2)
        for col in numeric_columns:
            total = df[col].sum()
            average_per_day = total / days
            
            # Creating three columns
            
            
            # Display each metric in its own card, vertically arranged by using the same column index for each metric
            with col1:
                st.metric(label=f"Total {col}", value=f"{total.round()}")
            with col2:
                st.metric(label=f"Average {col} per day", value=f"$ {average_per_day:.1f}")
        
        if 'Balance' in numeric_columns:
            with col1:
                st.metric(label="Opening Balance", value=f"{df['Balance'].iloc[0].round()}")
            with col2:
                st.metric(label="Closing Balance", value=f"{df['Balance'].iloc[-1].round()}")
        with col1:
                    st.metric(label="Total Transactions", value=f"{len(df)}")

        with col2:
                    st.metric(label="Number of Days", value=f" {days}")
        for col in numeric_columns:
            df[f'Cumulative {col}'] = df[col].cumsum()
        
        df_plot = df.set_index('Date')

        # Allow user to select which cumulative column to visualize
        selected_col = st.selectbox('Select a variable to inspect its trend:', ['Please Select'] + list(numeric_columns))

        # If a selection is made, display the trend for the selected column
        if selected_col != 'Please Select':
            # Subheader for the selected trend visualization
            st.subheader(f'{selected_col} Trend')
            st.line_chart(df_plot[selected_col], use_container_width=True)

        # Selection for further visualization
        # Allow the user to select a numeric column to visualize
        selected_col = st.radio('Select a variable to inspect its comulative trend:', numeric_columns)

        # Display trend based on user selection
        if selected_col:
            # Dynamically adjust subheader and chart based on selection
            st.subheader(f'{selected_col} Trend')
            
            # Display cumulative trend
            st.line_chart(df_plot[f'Cumulative {selected_col}'], use_container_width=True)
        df = df.reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_selected = st.date_input('Select Date', min_value=min_date, max_value=max_date, value=min_date)
        
        selected = df[df['Date'] == date_selected]
    
        # Display the filtered DataFrame
        st.dataframe(selected)

        # Identify all numeric columns in the DataFrame
        numeric_columns = selected.select_dtypes(include='number').columns

        # Dynamically sum values for each numeric column and display in horizontal cards
        col1, col2 = st.columns(2)
        for index, col in enumerate(numeric_columns):
            total_sum = selected[col].sum()
#            Display the sums in two columns; adjust as needed for more columns
            if index % 2 == 0:
                with col1:
                    st.metric(label=f"Total {col} on {date_selected}", value=f"$ {total_sum.round()}")
            else:
                with col2:
                    st.metric(label=f"Total {col} on {date_selected}", value=f"$ {total_sum.round()}")
    #    Monthly and Yearly Analysis
        df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%B')
        df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
        
        month_selected = st.selectbox('Select Month', df['Month'].unique())
        year_selected = st.selectbox('Select Year', df['Year'].unique())
        
        selected_month_year = df[(df['Month'] == month_selected) & (df['Year'] == year_selected)]
        st.dataframe(selected_month_year)
        for col in numeric_columns:
            total_sum = selected_month_year[col].sum()
            st.write(f"Total {col} in {month_selected} {year_selected}: {total_sum.round()}")
#######
       #st.header("Data Visualization")
        
        numeric_columns = df.select_dtypes(include=['number']).columns
        #selected_columns = st.multiselect("Select Columns To Plot", numeric_columns)

        # Define your columns
        st.title('Histogram with Date Selector')

        # Allow the user to select the date column and numeric column
        numeric_column = st.selectbox('Select Numeric Column:', options=df.select_dtypes(include=['float64', 'int']).columns)

        # Create a histogram of the selected numeric column over time
        if st.button('Plot Histogram'):
            fig = px.histogram(df, x='Date', y=numeric_column, histfunc='sum', title=f'Sum of {numeric_column} Over Time')
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig)

        st.title('Scatter plot with Date Selector')
        sel = st.multiselect("Select Columns for scatter plot", numeric_columns)

        if st.button('Scatter plot'):
    
            fig = px.scatter(df, x=sel[0], y=sel[1])
            st.plotly_chart(fig)
         
#######
        st.write("\n")
        st.subheader('Select a date range')
        start_range = df['Date'].iloc[0]
        end_range = df['Date'].iloc[-1]
        start_date = st.date_input('Start date', value=start_range)
        end_date = st.date_input('End date', value=end_range)
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        filtered_df = df.loc[mask]

        # Display the filtered DataFrame
        st.dataframe(filtered_df, use_container_width=True)

        # Dynamically sum values for each numeric column in the filtered DataFrame
        numeric_columns = filtered_df.select_dtypes(include='number').columns

        for col in numeric_columns:
            total_sum = filtered_df[col].sum()
            st.write(f'Total {col}: $ {total_sum}')


           # Let the user select one or more columns to group by
        group_columns = st.multiselect('Select detail(s) to group by:', options=df.columns)

        # Let the user select one or more numeric columns to aggregate
        agg_columns = st.multiselect('Select numeric column(s) to sum:', options=df.select_dtypes(include='number').columns)

        # Check if user has made selections before proceeding
        if group_columns and agg_columns:
            st.subheader('Total amount spent on each Detail')
            
            # Group by selected column(s) and sum selected numeric column(s)
            grouped_df = df.groupby(group_columns)[agg_columns].sum()
            
            # If multiple aggregation columns are selected, no need to sort_values
            if len(agg_columns) == 1:
                grouped_df = grouped_df.sort_values(by=agg_columns[0], ascending=False)
            
            # Display the aggregated DataFrame
            st.dataframe(grouped_df, use_container_width=True)
        else:
            st.write("Please select at least one detail and one numeric column to aggregate.")
       
        selected_column = st.selectbox('Select a numeric column to find the highest transaction:', options=df.select_dtypes(include='number').columns)

        if selected_column:
            st.subheader(f'Highest amount spent in one transaction for {selected_column}')
            
            # Find the index of the maximum value in the selected column
            max_value_index = df[selected_column].idxmax()
            
            # Display the row with the highest value for the selected column
            st.dataframe(df.loc[[max_value_index]], use_container_width=True)
        else:
            st.write("Please select a numeric column.")


        df['Date'] = pd.to_datetime(df['Date'])

      
        # Streamlit application setup
        import numpy as np
       # st.title('Monthly Average Value Calculator for Selected Date Range')
        # Corrected Function to identify if date is between the 22nd of the current month and the 1st of the next month
        def is_in_date_range(date):
            # Check if the date is on or after the 22nd
            if date.day >= 22:
                return True
            # Check if the date is the 1st of the month
            #elif date.day == 1:
             #   return True
            return False

        # Apply the corrected function to filter the DataFrame
        df['is_in_range'] = df['Date'].apply(is_in_date_range)
        df_filtered = df[df['is_in_range']]

        # Streamlit UI components
        st.title('Monthly Average Value for 22 - 31 for every month ')

        # Allow the user to select numeric columns for analysis
        selected_columns = st.multiselect('Select numeric columns:', df.select_dtypes(include=np.number).columns)

        if selected_columns:
            # Add 'Year' and 'Month' to the filtered DataFrame for grouping
            df_filtered['Year'] = df_filtered['Date'].dt.year
            df_filtered['Month'] = df_filtered['Date'].dt.month
            
            # Calculate the average for the selected columns, grouped by Year and Month
            averages = df_filtered.groupby(['Year', 'Month'])[selected_columns].mean().reset_index()
            
            # Display the averages table
            st.write('Average values for selected columns:', averages)
            
            # Visualization: Plotting the averages
            for column in selected_columns:
                fig = px.line(averages, x='Month', y=column, color='Year', markers=True,
                            labels={'Month': 'Month', column: f'Average {column}'},
                            title=f'Monthly Average of {column} Over Time')
                st.plotly_chart(fig)
        else:
            st.write("Please select at least one numeric column.")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
