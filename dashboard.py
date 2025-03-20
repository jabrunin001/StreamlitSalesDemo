"""
Sales Data Visualization Dashboard

This Streamlit application provides interactive visualizations of sales data
across multiple dimensions including time, products, regions, and customer segments.

Business Value:
- Provides immediate access to critical business metrics
- Enables data-driven decision making through interactive visualizations
- Automates reporting that would otherwise require manual data processing
- Creates a consistent view of business performance across the organization
- Enables self-service data exploration without technical skills
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar

# Set page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the data
@st.cache_data
def load_data():
    """
    Load and prepare the dataset for visualization.
    
    In a real application, this would connect to a database, API, or data warehouse.
    For this example, we'll generate synthetic sales data.
    
    Returns:
        pandas.DataFrame: The prepared sales dataset
    """
    # Generate dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 6, 30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define product categories and products
    categories = ['Electronics', 'Clothing', 'Home Goods', 'Sporting Goods', 'Books']
    products_by_category = {
        'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
        'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
        'Home Goods': ['Couch', 'Bed', 'Table', 'Chair', 'Lamp'],
        'Sporting Goods': ['Basketball', 'Tennis Racket', 'Bicycle', 'Treadmill', 'Weights'],
        'Books': ['Fiction', 'Non-Fiction', 'Biography', 'Cookbook', 'Self-Help']
    }
    
    # Define regions and stores
    regions = ['North', 'South', 'East', 'West', 'Central']
    stores_by_region = {
        'North': ['N001', 'N002', 'N003'],
        'South': ['S001', 'S002', 'S003', 'S004'],
        'East': ['E001', 'E002'],
        'West': ['W001', 'W002', 'W003'],
        'Central': ['C001', 'C002', 'C003', 'C004']
    }
    
    # Customer segments
    segments = ['New', 'Returning', 'Loyal', 'VIP']
    
    # Generate records
    np.random.seed(42)  # For reproducibility
    records = []
    
    # Seasonal patterns
    seasonality = {
        1: 0.8,   # January - post-holiday drop
        2: 0.7,   # February
        3: 0.9,   # March
        4: 1.0,   # April
        5: 1.1,   # May
        6: 1.2,   # June
        7: 1.3,   # July
        8: 1.2,   # August - back to school
        9: 1.0,   # September
        10: 1.1,  # October
        11: 1.4,  # November - holiday shopping
        12: 1.8   # December - holiday peak
    }
    
    # Category seasonality
    category_seasonality = {
        'Electronics': {11: 1.7, 12: 2.0},  # Holiday boost
        'Clothing': {3: 1.2, 4: 1.3, 8: 1.5, 9: 1.4},  # Spring and back-to-school
        'Home Goods': {1: 1.2, 5: 1.4, 6: 1.5},  # New Year and summer home improvement
        'Sporting Goods': {1: 1.3, 5: 1.5, 6: 1.6, 7: 1.7},  # New Year resolutions and summer
        'Books': {6: 1.3, 7: 1.4, 12: 1.5}  # Summer reading and holiday gifts
    }
    
    # Weekly patterns
    weekly_pattern = {
        0: 0.7,  # Monday
        1: 0.8,  # Tuesday
        2: 0.9,  # Wednesday
        3: 1.0,  # Thursday
        4: 1.3,  # Friday
        5: 1.5,  # Saturday
        6: 1.0   # Sunday
    }
    
    # Generate 10,000 sales records
    for _ in range(10000):
        # Pick a random date
        date = np.random.choice(dates)
        
        # Convert to pandas Timestamp
        date = pd.Timestamp(date)  # Ensure date is a pandas Timestamp
        
        # Apply seasonality
        month_factor = seasonality[date.month]
        weekday_factor = weekly_pattern[date.weekday()]
        
        # Pick category and product
        category = np.random.choice(categories)
        product = np.random.choice(products_by_category[category])
        
        # Apply category-specific seasonality
        category_month_factor = category_seasonality.get(category, {}).get(date.month, 1.0)
        
        # Pick region and store
        region = np.random.choice(regions)
        store = np.random.choice(stores_by_region[region])
        
        # Pick customer segment
        segment = np.random.choice(segments)
        
        # Generate sales quantity and price
        base_price = np.random.uniform(10, 500)  # Base price varies by product
        
        # Price ranges by category
        if category == 'Electronics':
            base_price = np.random.uniform(200, 1500)
        elif category == 'Clothing':
            base_price = np.random.uniform(20, 200)
        elif category == 'Home Goods':
            base_price = np.random.uniform(50, 1000)
        elif category == 'Sporting Goods':
            base_price = np.random.uniform(30, 800)
        elif category == 'Books':
            base_price = np.random.uniform(10, 50)
        
        # Apply minor price variation
        price = base_price * np.random.uniform(0.9, 1.1)
        
        # Calculate quantity and total
        quantity = np.random.randint(1, 5)
        
        # Apply all seasonality factors
        quantity = max(1, int(quantity * month_factor * weekday_factor * category_month_factor))
        
        # Calculate revenue
        revenue = price * quantity
        
        # Calculate profit (assume 30-50% margin)
        margin = np.random.uniform(0.3, 0.5)
        profit = revenue * margin
        
        # Create record
        record = {
            'date': date,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'weekday': date.weekday(),
            'weekday_name': calendar.day_name[date.weekday()],
            'category': category,
            'product': product,
            'region': region,
            'store': store,
            'customer_segment': segment,
            'quantity': quantity,
            'price': round(price, 2),
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'transaction_id': f"TX-{np.random.randint(10000, 99999)}"
        }
        
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Add month-year column for easier grouping
    df['month_year'] = df['date'].dt.strftime('%b %Y')
    
    print(df.head())  # Check the first few rows of the DataFrame
    
    return df

# Load the data
df = load_data()

# Create a function to format large numbers
def format_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"${num/1e9:.1f}B"
    elif num >= 1e6:
        return f"${num/1e6:.1f}M"
    elif num >= 1e3:
        return f"${num/1e3:.1f}K"
    else:
        return f"${num:.0f}"

# Dashboard title
st.title("📊 Sales Analytics Dashboard")
st.markdown("An interactive dashboard for analyzing sales performance across different dimensions.")

# Sidebar for filters
st.sidebar.header("Filters")

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Handle single date selection
if len(date_range) == 1:
    date_range = [date_range[0], max_date]

# Product category filter
categories = ['All'] + sorted(df['category'].unique().tolist())
selected_category = st.sidebar.selectbox("Product Category", categories)

# Region filter
regions = ['All'] + sorted(df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions)

# Customer segment filter
segments = ['All'] + sorted(df['customer_segment'].unique().tolist())
selected_segment = st.sidebar.selectbox("Customer Segment", segments)

# Apply filters
filtered_df = df.copy()

# Date filter
filtered_df = filtered_df[(filtered_df['date'].dt.date >= date_range[0]) & 
                         (filtered_df['date'].dt.date <= date_range[1])]

# Category filter
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]

# Region filter
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]

# Segment filter
if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['customer_segment'] == selected_segment]

# Calculate key metrics
total_revenue = filtered_df['revenue'].sum()
total_profit = filtered_df['profit'].sum()
total_orders = len(filtered_df)
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0

# Display key metrics
st.subheader("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Revenue", format_number(total_revenue))

with col2:
    st.metric("Total Profit", format_number(total_profit))

with col3:
    st.metric("Total Orders", f"{total_orders:,}")

with col4:
    st.metric("Avg. Order Value", format_number(avg_order_value))

with col5:
    st.metric("Profit Margin", f"{profit_margin:.1f}%")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Time Analysis", "Product Analysis", "Regional Analysis", "Customer Analysis"])

with tab1:
    st.subheader("Sales Performance Over Time")
    
    # Time granularity selector
    time_granularity = st.radio(
        "Select Time Granularity",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True
    )
    
    # Aggregate data based on selected granularity
    if time_granularity == "Daily":
        time_df = filtered_df.groupby('date').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        time_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
        x_axis = 'date'
        
    elif time_granularity == "Weekly":
        filtered_df['week'] = filtered_df['date'].dt.isocalendar().week
        filtered_df['year'] = filtered_df['date'].dt.isocalendar().year
        filtered_df['week_year'] = filtered_df['year'].astype(str) + '-W' + filtered_df['week'].astype(str).str.zfill(2)
        
        time_df = filtered_df.groupby('week_year').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count',
            'date': 'min'  # Use the first day of each week for plotting
        }).reset_index()
        time_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
        time_df.sort_values('date', inplace=True)
        x_axis = 'week_year'
        
    else:  # Monthly
        filtered_df['month_year'] = filtered_df['date'].dt.strftime('%b %Y')
        filtered_df['month_year_sort'] = filtered_df['date'].dt.strftime('%Y-%m')
        
        time_df = filtered_df.groupby(['month_year', 'month_year_sort']).agg({
            'revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        time_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
        time_df.sort_values('month_year_sort', inplace=True)
        x_axis = 'month_year'
    
    # Metric to visualize
    time_metric = st.selectbox(
        "Select Metric to Visualize",
        options=["Revenue", "Profit", "Orders", "Revenue vs Profit"],
        index=0
    )
    
    # Create the visualization
    if time_metric == "Revenue vs Profit":
        # Create a dual-axis chart with revenue and profit
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=time_df[x_axis],
                y=time_df['revenue'],
                name='Revenue',
                marker_color='#1f77b4'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_df[x_axis],
                y=time_df['profit'],
                name='Profit',
                marker_color='#ff7f0e',
                mode='lines+markers'
            )
        )
        
        fig.update_layout(
            title='Revenue vs Profit Over Time',
            xaxis_title='Time Period',
            yaxis_title='Amount ($)',
            legend_title='Metric',
            hovermode='x unified',
            height=500
        )
        
        # If we have many data points, rotate x-axis labels
        if len(time_df) > 10:
            fig.update_layout(
                xaxis=dict(
                    tickangle=-45,
                    tickmode='auto',
                    nticks=20
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Single metric visualization
        metric_column = time_metric.lower()
        
        fig = px.line(
            time_df,
            x=x_axis,
            y=metric_column,
            title=f"{time_metric} Over Time",
            markers=True
        )
        
        # Customize the chart
        fig.update_layout(
            xaxis_title='Time Period',
            yaxis_title=f"{time_metric} ({('$' if metric_column != 'orders' else '')})",
            hovermode='x unified',
            height=500
        )
        
        # If we have many data points, rotate x-axis labels
        if len(time_df) > 10:
            fig.update_layout(
                xaxis=dict(
                    tickangle=-45,
                    tickmode='auto',
                    nticks=20
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add a histogram of sales by day of week
    st.subheader("Sales by Day of Week")
    
    weekday_df = filtered_df.groupby(['weekday', 'weekday_name']).agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    weekday_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Create a custom sort order for days of the week
    weekday_order = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    weekday_df['weekday_order'] = weekday_df['weekday_name'].map(weekday_order)
    weekday_df.sort_values('weekday_order', inplace=True)
    
    # Create the visualization
    weekday_metric = st.selectbox(
        "Select Metric for Day of Week Analysis",
        options=["Revenue", "Profit", "Orders"],
        index=0,
        key="weekday_metric"
    )
    
    weekday_chart = px.bar(
        weekday_df,
        x='weekday_name',
        y=weekday_metric.lower(),
        title=f"{weekday_metric} by Day of Week",
        color='weekday_name',
        labels={'weekday_name': 'Day of Week', weekday_metric.lower(): weekday_metric}
    )
    
    weekday_chart.update_layout(
        xaxis_title='Day of Week',
        yaxis_title=f"{weekday_metric} ({('$' if weekday_metric.lower() != 'orders' else '')})",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(weekday_chart, use_container_width=True)

with tab2:
    st.subheader("Product Performance Analysis")
    
    # Category performance
    category_df = filtered_df.groupby('category').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    category_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    category_df.sort_values('revenue', ascending=False, inplace=True)
    
    # Product metric selection
    product_metric = st.selectbox(
        "Select Metric for Product Analysis",
        options=["Revenue", "Profit", "Orders", "Profit Margin"],
        index=0
    )
    
    if product_metric == "Profit Margin":
        category_df['profit_margin'] = (category_df['profit'] / category_df['revenue']) * 100
        y_column = 'profit_margin'
        y_title = "Profit Margin (%)"
    else:
        y_column = product_metric.lower()
        y_title = f"{product_metric} ({('$' if y_column != 'orders' else '')})"
    
    # Category chart
    fig = px.bar(
        category_df,
        x='category',
        y=y_column,
        title=f"Category Performance by {product_metric}",
        color='category',
        text_auto='.2s' if y_column != 'profit_margin' else '.1f'
    )
    
    fig.update_layout(
        xaxis_title='Product Category',
        yaxis_title=y_title,
        showlegend=False,
        height=400
    )
    
    # Add % sign to profit margin values
    if product_metric == "Profit Margin":
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top products analysis
    st.subheader("Top Products")
    
    # Number of top products to show
    top_n = st.slider("Number of Products to Display", min_value=5, max_value=20, value=10)
    
    # Group by product
    product_df = filtered_df.groupby(['product', 'category']).agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    product_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Calculate profit margin
    product_df['profit_margin'] = (product_df['profit'] / product_df['revenue']) * 100
    
    # Sort and get top N
    if product_metric == "Profit Margin":
        product_df = product_df.sort_values('profit_margin', ascending=False).head(top_n)
    else:
        product_df = product_df.sort_values(y_column, ascending=False).head(top_n)
    
    # Create visualization
    product_fig = px.bar(
        product_df,
        x='product',
        y=y_column,
        title=f"Top {top_n} Products by {product_metric}",
        color='category',
        text_auto='.2s' if y_column != 'profit_margin' else '.1f',
        hover_data=['category', 'revenue', 'profit', 'orders', 'profit_margin']
    )
    
    product_fig.update_layout(
        xaxis_title='Product',
        yaxis_title=y_title,
        legend_title='Category',
        height=500,
        xaxis={'categoryorder':'total descending'}
    )
    
    # Add % sign to profit margin values
    if product_metric == "Profit Margin":
        product_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    st.plotly_chart(product_fig, use_container_width=True)
    
    # Product trends over time
    st.subheader("Product Category Trends")
    
    # Group by month and category
    trend_df = filtered_df.copy()
    trend_df['month_year'] = trend_df['date'].dt.strftime('%b %Y')
    trend_df['month_year_sort'] = trend_df['date'].dt.strftime('%Y-%m')
    
    category_trend = trend_df.groupby(['month_year', 'month_year_sort', 'category']).agg({
        'revenue': 'sum'
    }).reset_index()
    
    # Sort by month-year
    category_trend = category_trend.sort_values('month_year_sort')
    
    # Create line chart for category trends
    trend_fig = px.line(
        category_trend,
        x='month_year',
        y='revenue',
        color='category',
        title='Revenue Trends by Product Category',
        markers=True
    )
    
    trend_fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Revenue ($)',
        legend_title='Category',
        height=500,
        hovermode='x unified'
    )
    
    # If we have many data points, rotate x-axis labels
    if len(category_trend['month_year'].unique()) > 6:
        trend_fig.update_layout(
            xaxis=dict(
                tickangle=-45,
                tickmode='auto',
                nticks=20
            )
        )
    
    st.plotly_chart(trend_fig, use_container_width=True)

with tab3:
    st.subheader("Regional Performance Analysis")
    
    # Region performance
    region_df = filtered_df.groupby('region').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    region_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Calculate profit margin
    region_df['profit_margin'] = (region_df['profit'] / region_df['revenue']) * 100
    
    # Regional metric selection
    regional_metric = st.selectbox(
        "Select Metric for Regional Analysis",
        options=["Revenue", "Profit", "Orders", "Profit Margin"],
        index=0,
        key="regional_metric"
    )
    
    if regional_metric == "Profit Margin":
        y_column = 'profit_margin'
        y_title = "Profit Margin (%)"
    else:
        y_column = regional_metric.lower()
        y_title = f"{regional_metric} ({('$' if y_column != 'orders' else '')})"
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Region bar chart
        region_fig = px.bar(
            region_df,
            x='region',
            y=y_column,
            title=f"Regional Performance by {regional_metric}",
            color='region',
            text_auto='.2s' if y_column != 'profit_margin' else '.1f'
        )
        
        region_fig.update_layout(
            xaxis_title='Region',
            yaxis_title=y_title,
            showlegend=False,
            height=400
        )
        
        # Add % sign to profit margin values
        if regional_metric == "Profit Margin":
            region_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        st.plotly_chart(region_fig, use_container_width=True)
    
    with col2:
        # Region pie chart
        pie_fig = px.pie(
            region_df,
            values=y_column,
            names='region',
            title=f"Regional Distribution of {regional_metric}",
            hole=0.4
        )
        
        pie_fig.update_layout(
            legend_title='Region',
            height=400
        )
        
        # Add percentage to labels
        pie_fig.update_traces(textinfo='percent+label')
        
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Store performance within each region
    st.subheader("Store Performance by Region")
    
    # Group by region and store
    store_df = filtered_df.groupby(['region', 'store']).agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    store_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Calculate profit margin
    store_df['profit_margin'] = (store_df['profit'] / store_df['revenue']) * 100
    
    
    # Create bar chart grouped by region
    store_fig = px.bar(
        store_df,
        x='store',
        y=y_column,
        color='region',
        title=f"Store Performance by {regional_metric}",
        barmode='group',
        text_auto='.2s' if y_column != 'profit_margin' else '.1f'
    )
    
    store_fig.update_layout(
        xaxis_title='Store',
        yaxis_title=y_title,
        legend_title='Region',
        height=500
    )
    
    # Add % sign to profit margin values
    if regional_metric == "Profit Margin":
        store_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    st.plotly_chart(store_fig, use_container_width=True)
    
    # Region and category heatmap
    st.subheader("Region by Category Performance")
    
    # Create pivot table of region by category
    region_category = filtered_df.pivot_table(
        index='region',
        columns='category',
        values=y_column.replace('_', ' ').title(),
        aggfunc='sum'
    )
    
    # Create heatmap
    heatmap_fig = px.imshow(
        region_category,
        labels=dict(x="Category", y="Region", color=y_title),
        title=f"Heatmap: {regional_metric} by Region and Category",
        text_auto='.2s' if y_column != 'profit_margin' else '.1f',
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    
    heatmap_fig.update_layout(
        height=400
    )
    
    st.plotly_chart(heatmap_fig, use_container_width=True)

with tab4:
    st.subheader("Customer Segment Analysis")
    
    # Segment performance
    segment_df = filtered_df.groupby('customer_segment').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    segment_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Calculate profit margin and average order value
    segment_df['profit_margin'] = (segment_df['profit'] / segment_df['revenue']) * 100
    
    segment_df['avg_order_value'] = segment_df['revenue'] / segment_df['orders']
    
    # Customer metric selection
    customer_metric = st.selectbox(
        "Select Metric for Customer Analysis",
        options=["Revenue", "Profit", "Orders", "Profit Margin", "Avg Order Value"],
        index=0
    )
    
    if customer_metric == "Profit Margin":
        y_column = 'profit_margin'
        y_title = "Profit Margin (%)"
    elif customer_metric == "Avg Order Value":
        y_column = 'avg_order_value'
        y_title = "Average Order Value ($)"
    else:
        y_column = customer_metric.lower()
        y_title = f"{customer_metric} ({'' if y_column in ['orders', 'profit_margin'] else ''})"
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment bar chart
        segment_fig = px.bar(
            segment_df,
            x='customer_segment',
            y=y_column,
            title=f"Customer Segment Performance by {customer_metric}",
            color='customer_segment',
            text_auto='.2s' if y_column not in ['profit_margin'] else '.1f'
        )
        
        segment_fig.update_layout(
            xaxis_title='Customer Segment',
            yaxis_title=y_title,
            showlegend=False,
            height=400
        )
        
        # Add % sign to profit margin values
        if customer_metric == "Profit Margin":
            segment_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        st.plotly_chart(segment_fig, use_container_width=True)
    
    with col2:
        # Segment pie chart
        pie_fig = px.pie(
            segment_df,
            values=y_column,
            names='customer_segment',
            title=f"Customer Segment Distribution of {customer_metric}",
            hole=0.4
        )
        
        pie_fig.update_layout(
            legend_title='Customer Segment',
            height=400
        )
        
        # Add percentage to labels
        pie_fig.update_traces(textinfo='percent+label')
        
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Customer segment trends over time
    st.subheader("Customer Segment Trends")
    
    # Group by month and segment
    segment_trend_df = filtered_df.copy()
    segment_trend_df['month_year'] = segment_trend_df['date'].dt.strftime('%b %Y')
    segment_trend_df['month_year_sort'] = segment_trend_df['date'].dt.strftime('%Y-%m')
    
    segment_trend = segment_trend_df.groupby(['month_year', 'month_year_sort', 'customer_segment']).agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    segment_trend.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Calculate derived metrics
    segment_trend['profit_margin'] = (segment_trend['profit'] / segment_trend['revenue']) * 100
    segment_trend['avg_order_value'] = segment_trend['revenue'] / segment_trend['orders']
    
    # Sort by month-year
    segment_trend = segment_trend.sort_values('month_year_sort')
    
    # Create line chart for segment trends
    trend_metric = st.selectbox(
        "Select Metric for Trend Analysis",
        options=["Revenue", "Profit", "Orders", "Profit Margin", "Avg Order Value"],
        index=0,
        key="trend_metric"
    )
    
    # Set y-column based on selected metric
    if trend_metric == "Profit Margin":
        trend_y = 'profit_margin'
        trend_y_title = "Profit Margin (%)"
    elif trend_metric == "Avg Order Value":
        trend_y = 'avg_order_value'
        trend_y_title = "Average Order Value ($)"
    else:
        trend_y = trend_metric.lower()
        trend_y_title = f"{trend_metric} ({'' if trend_y not in ['orders', 'profit_margin'] else ''})"
    
    # Create visualization
    trend_fig = px.line(
        segment_trend,
        x='month_year',
        y=trend_y,
        color='customer_segment',
        title=f'{trend_metric} Trends by Customer Segment',
        markers=True
    )
    
    trend_fig.update_layout(
        xaxis_title='Month',
        yaxis_title=trend_y_title,
        legend_title='Customer Segment',
        height=500,
        hovermode='x unified'
    )
    
    # If we have many data points, rotate x-axis labels
    if len(segment_trend['month_year'].unique()) > 6:
        trend_fig.update_layout(
            xaxis=dict(
                tickangle=-45,
                tickmode='auto',
                nticks=20
            )
        )
    
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Segment by category analysis
    st.subheader("Customer Segment Performance by Product Category")
    
    # Create a pivot table of segment by category
    segment_category = filtered_df.pivot_table(
        index='customer_segment',
        columns='category',
        values='revenue',
        aggfunc='sum'
    )
    
    # Create heatmap
    heatmap_fig = px.imshow(
        segment_category,
        labels=dict(x="Product Category", y="Customer Segment", color=y_title),
        title=f"Heatmap: {customer_metric} by Segment and Category",
        text_auto='.2s' if y_column not in ['profit_margin'] else '.1f',
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    
    heatmap_fig.update_layout(
        height=400
    )
    
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Add a collapsible section with raw data 
st.subheader("Raw Data Explorer")
with st.expander("View and Export Raw Data"):
    # Add a search box
    search_term = st.text_input("Search in data (product, category, store, etc.)")
    
    # Filter based on search term if provided
    if search_term:
        search_results = filtered_df[
            filtered_df.apply(
                lambda row: any(
                    search_term.lower() in str(value).lower() 
                    for value in row.values
                ), 
                axis=1
            )
        ]
        st.dataframe(search_results)
        
        # Download button for search results
        csv = search_results.to_csv(index=False)
        st.download_button(
            label="Download Search Results as CSV",
            data=csv,
            file_name="sales_data_search_results.csv",
            mime="text/csv"
        )
    else:
        # Show the filtered data
        st.dataframe(filtered_df)
        
        # Download button for filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="sales_data_filtered.csv",
            mime="text/csv"
        )

# Add a footer with information about the dashboard
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>Sales Analytics Dashboard | Built with Streamlit | Data last updated: June 30, 2023</p>
</div>
""", unsafe_allow_html=True)

st.write(filtered_df.columns)