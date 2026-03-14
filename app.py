import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Load the data
@st.cache_data
def load_data():
    try:
        # Use read_excel for .xlsx files
        data = pd.read_excel(r"C:\Users\priya\Downloads\archive (3)\Online Retail.xlsx")
        
        # Data cleaning and preprocessing
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        data['Month'] = data['InvoiceDate'].dt.to_period('M').astype(str)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(layout="wide", page_title="Retail Analytics Dashboard")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No data available. Please check your Excel file.")
        return
    
    # Main Dashboard Title
    st.title('🛍️ Comprehensive Retail Analytics Dashboard')
    
    # Sidebar Filters
    st.sidebar.header('🔍 Dashboard Filters')
    
    # Country Filter
    countries = st.sidebar.multiselect(
        'Select Countries', 
        options=df['Country'].unique(),
        default=df['Country'].unique()
    )
    
    # Date Range Filter
    date_range = st.sidebar.date_input(
        'Select Date Range',
        value=(df['InvoiceDate'].min().date(), df['InvoiceDate'].max().date())
    )
    
    # Filter Data
    filtered_df = df[
        (df['Country'].isin(countries)) & 
        (df['InvoiceDate'].dt.date >= pd.Timestamp(date_range[0]).date()) & 
        (df['InvoiceDate'].dt.date <= pd.Timestamp(date_range[1]).date())
    ]
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Transactions', filtered_df['InvoiceNo'].nunique())
    with col2:
        st.metric('Total Products', filtered_df['StockCode'].nunique())
    with col3:
        st.metric('Total Revenue', f'£{filtered_df["TotalPrice"].sum():,.2f}')
    with col4:
        st.metric('Avg Transaction Value', f'£{filtered_df["TotalPrice"].mean():,.2f}')
    
    # Tabs for Detailed Analytics
    tab1, tab2, tab3, tab4 = st.tabs([
        '📊 Sales Overview', 
        '🏆 Product Performance', 
        '🌍 Geographic Insights', 
        '📈 Advanced Analytics'
    ])
    
    with tab1:
        # Sales by Month Line Chart
        st.subheader('Monthly Sales Trend')
        monthly_sales = filtered_df.groupby('Month')['TotalPrice'].sum().reset_index()
        
        # Convert Month column to datetime for proper sorting
        monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])
        monthly_sales = monthly_sales.sort_values('Month')
        
        fig_monthly = px.line(
            monthly_sales, 
            x='Month', 
            y='TotalPrice', 
            title='Monthly Sales Trend',
            labels={'TotalPrice': 'Total Sales (£)', 'Month': 'Month'}
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Revenue by Product Category
        st.subheader('Revenue by Product Category')
        category_revenue = filtered_df.groupby('Description')['TotalPrice'].sum().nlargest(10)
        fig_category = px.bar(
            category_revenue,
            x=category_revenue.index, 
            y=category_revenue.values,
            title='Top 10 Product Categories by Revenue',
            labels={'x': 'Product', 'y': 'Total Revenue (£)'}
        )
        st.plotly_chart(fig_category, use_container_width=True)
    
    with tab2:
        # Top Selling Products
        st.subheader('Top Selling Products by Quantity')
        top_products_qty = filtered_df.groupby('Description')['Quantity'].sum().nlargest(10)
        fig_top_products_qty = px.bar(
            top_products_qty,
            x=top_products_qty.index, 
            y=top_products_qty.values,
            title='Top 10 Products by Quantity Sold',
            labels={'x': 'Product', 'y': 'Quantity'}
        )
        st.plotly_chart(fig_top_products_qty, use_container_width=True)
        
        # Top Products by Revenue
        st.subheader('Top Products by Revenue')
        top_products_revenue = filtered_df.groupby('Description')['TotalPrice'].sum().nlargest(10)
        fig_top_products_rev = px.bar(
            top_products_revenue,
            x=top_products_revenue.index, 
            y=top_products_revenue.values,
            title='Top 10 Products by Total Revenue',
            labels={'x': 'Product', 'y': 'Total Revenue (£)'}
        )
        st.plotly_chart(fig_top_products_rev, use_container_width=True)
    
    with tab3:
        # Sales by Country
        st.subheader('Sales Distribution by Country')
        country_sales = filtered_df.groupby('Country')['TotalPrice'].sum()
        fig_country_sales = px.pie(
            values=country_sales.values, 
            names=country_sales.index, 
            title='Sales Distribution Across Countries'
        )
        st.plotly_chart(fig_country_sales, use_container_width=True)
        
        # Transactions by Country
        st.subheader('Transactions by Country')
        country_transactions = filtered_df.groupby('Country')['InvoiceNo'].nunique()
        fig_country_trans = px.bar(
            x=country_transactions.index, 
            y=country_transactions.values,
            title='Number of Transactions by Country',
            labels={'x': 'Country', 'y': 'Number of Transactions'}
        )
        st.plotly_chart(fig_country_trans, use_container_width=True)
    
    with tab4:
        # Seasonality Analysis
        st.subheader('Seasonality Analysis')
        filtered_df['Month_Num'] = filtered_df['InvoiceDate'].dt.month
        seasonal_sales = filtered_df.groupby('Month_Num')['TotalPrice'].sum()
        fig_seasonality = px.bar(
            x=seasonal_sales.index, 
            y=seasonal_sales.values,
            title='Sales by Month',
            labels={'x': 'Month', 'y': 'Total Sales (£)'}
        )
        st.plotly_chart(fig_seasonality, use_container_width=True)
        
        # Product Diversity
        st.subheader('Product Diversity')
        product_diversity = filtered_df.groupby('Country')['StockCode'].nunique()
        fig_product_diversity = px.bar(
            x=product_diversity.index, 
            y=product_diversity.values,
            title='Number of Unique Products by Country',
            labels={'x': 'Country', 'y': 'Unique Products'}
        )
        st.plotly_chart(fig_product_diversity, use_container_width=True)

if __name__ == '__main__':
    main()