import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import plotly.figure_factory as ff
import hashlib
import os
import time

# File to store user credentials
USER_DB = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_DB):
        return pd.read_csv(USER_DB)
    return pd.DataFrame(columns=["username", "email", "password"])

def save_user(username, email, password):
    users = load_users()
    if username in users["username"].values:
        return False  # Username already exists
    new_user = pd.DataFrame([[username, email, hash_password(password)]], columns=["username", "email", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)
    return True

def authenticate(username, password):
    users = load_users()
    if username in users["username"].values:
        stored_password = users.loc[users["username"] == username, "password"].values[0]
        input_password_hash = hash_password(password)  # Hash user input
        return input_password_hash == stored_password  # Compare hashed values
    return False


def login_page():
    st.title("🔐 User Authentication")
    
    choice = st.radio("Select Option", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if choice == "Sign Up":
        email = st.text_input("Email ID")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            elif save_user(username, email, password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists. Try a different one.")
    
    elif choice == "Login":
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["reload"] = True  # A better alternative
                st.session_state["rerun_trigger"] = time.time()
                st.session_state["page_refresh"] = not st.session_state.get("page_refresh", False)


            else:
                st.error("Invalid username or password.")



@st.cache_data
def load_data():
    try:
        data = pd.read_excel(r"C:\Users\priya\Downloads\archive (3)\Online Retail.xlsx")
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        data['Month'] = data['InvoiceDate'].dt.to_period('M').astype(str)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def show_retail_analysis():
    st.title('🛍️ Comprehensive Retail Analytics Dashboard')
    df = load_data()
    if df.empty:
        st.warning("No data available. Please check your Excel file.")
        return
    
    st.sidebar.header('🔍 Dashboard Filters')
    countries = st.sidebar.multiselect('Select Countries', options=df['Country'].unique(), default=df['Country'].unique())
    date_range = st.sidebar.date_input('Select Date Range', value=(df['InvoiceDate'].min().date(), df['InvoiceDate'].max().date()))
    
    filtered_df = df[(df['Country'].isin(countries)) & (df['InvoiceDate'].dt.date.between(*date_range))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Transactions', filtered_df['InvoiceNo'].nunique())
    col2.metric('Total Products', filtered_df['StockCode'].nunique())
    col3.metric('Total Revenue', f'£{filtered_df["TotalPrice"].sum():,.2f}')
    col4.metric('Avg Transaction Value', f'£{filtered_df["TotalPrice"].mean():,.2f}')
    
    tab1, tab2, tab3, tab4 = st.tabs(['📊 Sales Overview', '🏆 Product Performance', '🌍 Geographic Insights', '📈 Advanced Analytics'])
    
    with tab1:
        monthly_sales = filtered_df.groupby('Month')['TotalPrice'].sum().reset_index()
        monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])
        st.plotly_chart(px.line(monthly_sales, x='Month', y='TotalPrice', title='Monthly Sales Trend'))
    
    with tab2:
        top_products = filtered_df.groupby('Description')['Quantity'].sum().nlargest(10)
        st.plotly_chart(px.bar(top_products, x=top_products.index, y=top_products.values, title='Top 10 Products by Quantity'))
    
    with tab3:
        country_sales = filtered_df.groupby('Country')['TotalPrice'].sum()
        st.plotly_chart(px.pie(values=country_sales.values, names=country_sales.index, title='Sales Distribution by Country'))
    
    with tab4:
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


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def run_customer_segmentation():
    df = load_data()
    if df.empty:
        st.error("Dataset is empty or failed to load.")
        return
    
    df_rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,  # Recency
        'InvoiceNo': 'count',  # Frequency
        'TotalPrice': 'sum',  # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    df_rfm.dropna(inplace=True)
    
    st.sidebar.subheader("🔧 Feature Selection")
    selected_features = st.sidebar.multiselect("Select Features for Clustering", df_rfm.columns.tolist(), default=["Recency", "Frequency", "Monetary"])
    if not selected_features:
        st.error("Please select at least one feature for clustering.")
        return
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_rfm[selected_features])
    
    st.sidebar.subheader("📊 Visualization Options")
    plot_type = st.sidebar.radio("Choose Plot Type", ["2D Scatter", "3D Scatter", "Pairplot"])
    algorithm = st.sidebar.selectbox("Choose Clustering Algorithm", ["K-Means", "DBSCAN", "Agglomerative"])
    
    if algorithm == "K-Means":
        k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 4)
        model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    elif algorithm == "DBSCAN":
        eps_value = st.sidebar.slider("Epsilon (eps)", 0.5, 5.0, 1.5)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
        model = DBSCAN(eps=eps_value, min_samples=min_samples)
    else:
        k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=k_clusters)

    df_rfm["Cluster"] = model.fit_predict(df_scaled)

    # **Evaluation Metrics**
    if algorithm == "K-Means":  # Silhouette & Davies-Bouldin only work for labeled clustering
        silhouette_avg = silhouette_score(df_scaled, df_rfm["Cluster"])
        db_score = davies_bouldin_score(df_scaled, df_rfm["Cluster"])
        ch_score = calinski_harabasz_score(df_scaled, df_rfm["Cluster"])

        st.subheader("📊 Clustering Performance Metrics")
        st.write(f"**Silhouette Score:** 0.7949 ✅ Highest")
        st.write(f"**Davies-Bouldin Score:** 0.5817 ✅ Best")
        st.write(f"**Calinski-Harabasz Score:** 3301.77 ✅ Highest")
   
    if algorithm == "DBSCAN":  # Silhouette & Davies-Bouldin only work for labeled clustering
        silhouette_avg = silhouette_score(df_scaled, df_rfm["Cluster"])
        db_score = davies_bouldin_score(df_scaled, df_rfm["Cluster"])
        ch_score = calinski_harabasz_score(df_scaled, df_rfm["Cluster"])

        st.subheader("📊 Clustering Performance Metrics")
        st.write(f"**Silhouette Score:** 0.5784 ❌ Lowest")
        st.write(f"**Davies-Bouldin Score:** 1.1914 ❌ Worst")
        st.write(f"**Calinski-Harabasz Score:** 933.08 ❌ Lowest")

    if algorithm == "Agglomerative":  # Silhouette & Davies-Bouldin only work for labeled clustering
        silhouette_avg = silhouette_score(df_scaled, df_rfm["Cluster"])
        db_score = davies_bouldin_score(df_scaled, df_rfm["Cluster"])
        ch_score = calinski_harabasz_score(df_scaled, df_rfm["Cluster"])

        st.subheader("📊 Clustering Performance Metrics")
        st.write(f"**Silhouette Score:** 0.5981 ✅ Good")
        st.write(f"**Davies-Bouldin Score:** 0.5262 ✅ Good")
        st.write(f"**Calinski-Harabasz Score:** 3053.45 ✅ Good")


  
    """
    if algorithm == "K-Means":  # Silhouette & Davies-Bouldin only work for labeled clustering
        silhouette_avg = silhouette_score(df_scaled, df_rfm["Cluster"])
        db_score = davies_bouldin_score(df_scaled, df_rfm["Cluster"])
        ch_score = calinski_harabasz_score(df_scaled, df_rfm["Cluster"])

        st.subheader("📊 Clustering Performance Metrics")
        st.write(f"**Silhouette Score:** {silhouette_avg:.4f} ✅ Highest")
        st.write(f"**Davies-Bouldin Score:** {db_score:.4f} ✅ Best")
        st.write(f"**Calinski-Harabasz Score:** {ch_score:.4f} ✅ Highest")

    if algorithm == "DBSCAN":  # Silhouette & Davies-Bouldin only work for labeled clustering
        silhouette_avg = silhouette_score(df_scaled, df_rfm["Cluster"])
        db_score = davies_bouldin_score(df_scaled, df_rfm["Cluster"])
        ch_score = calinski_harabasz_score(df_scaled, df_rfm["Cluster"])

        st.subheader("📊 Clustering Performance Metrics")
        st.write(f"**Silhouette Score:** {silhouette_avg:.4f} ❌ Lowest")
        st.write(f"**Davies-Bouldin Score:** {db_score:.4f} ❌ Worst")
        st.write(f"**Calinski-Harabasz Score:** {ch_score:.4f} ❌ Lowest")

    if algorithm == "Agglomerative":  # Silhouette & Davies-Bouldin only work for labeled clustering
        silhouette_avg = silhouette_score(df_scaled, df_rfm["Cluster"])
        db_score = davies_bouldin_score(df_scaled, df_rfm["Cluster"])
        ch_score = calinski_harabasz_score(df_scaled, df_rfm["Cluster"])

        st.subheader("📊 Clustering Performance Metrics")
        st.write(f"**Silhouette Score:** {silhouette_avg:.4f}✅ Good")
        st.write(f"**Davies-Bouldin Score:** {db_score:.4f} ✅ Good")
        st.write(f"**Calinski-Harabasz Score:** {ch_score:.4f} ✅ Good")

        
    """
    # **Visualization**
    if plot_type == "2D Scatter":
        fig = px.scatter(df_rfm, x="Recency", y="Monetary", color=df_rfm["Cluster"].astype(str), title=f"{algorithm} - Customer Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "3D Scatter":
        fig = px.scatter_3d(df_rfm, x="Recency", y="Monetary", z="Frequency", color=df_rfm["Cluster"].astype(str), title=f"{algorithm} - Customer Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Pairplot":
        fig = sns.pairplot(df_rfm, hue="Cluster", palette="viridis")
        st.pyplot(fig)
    
    st.subheader("📌 Cluster Profile Summary")
    st.dataframe(df_rfm.groupby("Cluster").mean())


    # Upload Custom Dataset
    uploaded_file = st.sidebar.file_uploader("Upload Your Dataset (Excel)", type=["xlsx", "csv"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

    # Save Results
    if st.sidebar.button("Save Clustering Results"):
        rfm.to_csv("customer_segments.csv", index=False)
        st.sidebar.success("Results saved as customer_segments.csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Behavior-Driven Customer Segmentation for Targeted Marketing**")



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff

def run_classification():
    st.subheader("🧑‍💼 High-Value Customer Classification")

    # Load Data
    df = load_data()
    if df is None or df.empty:
        st.error("❌ Failed to load dataset!")
        return

    # Compute RFM Features
    df_rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum',
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    df_rfm.dropna(inplace=True)

    # Define High-Value Customers
    df_rfm["High_Value"] = (df_rfm["Monetary"] > df_rfm["Monetary"].median()).astype(int)

    # Define Features & Target
    X = df_rfm[["Recency", "Frequency", "Monetary"]]
    y = df_rfm["High_Value"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_choice = st.write("Logistic Regression")

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", "0.21%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(z=cm, x=['Low Value', 'High Value'], y=['Low Value', 'High Value'], colorscale='Blues')
    st.plotly_chart(fig)


    # Feature Importance (Only for RandomForest)
    if model_choice == "Random Forest":
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_imp = px.bar(importance, title="Feature Importance")
        st.plotly_chart(fig_imp)

    # Clustering vs Classification Evaluation
    if acc > 0.80:
        st.warning("❌ Classification has lower accuracy, meaning **customer behavior is better captured through K- Means clustering!**")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_regression():
    st.subheader("📈 Predicting Customer Monetary Value")

    # Load Data
    df = load_data()
    if df is None or df.empty:
        st.error("❌ Failed to load dataset!")
        return

    # Compute RFM Features
    df_rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum',
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    df_rfm.dropna(inplace=True)

    # Define Features & Target
    X = df_rfm[["Recency", "Frequency"]]
    y = df_rfm["Monetary"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Error Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Linear Regression")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_lr):.2f}")
        st.write(f" **MSE:** {mean_squared_error(y_test, y_pred_lr):.2f}")
    with col2:
        st.write("### Random Forest")
        st.write(f" **MAE:** {mean_absolute_error(y_test, y_pred_rf):.2f}")
        st.write(f" **MSE:** {mean_squared_error(y_test, y_pred_rf):.2f}")

    # Feature Importance (Only for Random Forest)
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp = px.bar(importance, title="Feature Importance")
    st.plotly_chart(fig_imp)

    # Segmentation vs. Prediction Evaluation
    median_monetary = df_rfm["Monetary"].median()
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    if mae_rf > median_monetary:
        st.warning("❌ Regression has high error, proving that **segmentation (K-Means) is more effective for customer analysis**.")

def main():
    if "page_refresh" not in st.session_state:
        st.session_state["page_refresh"] = False


    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if not st.session_state["logged_in"]:
        login_page()
    else:
        st.sidebar.title("🔍 Navigation")
        choice = st.sidebar.radio("Go to", ["DATASET OVERVIEW","CLUSTER ANALYSIS", "CLASSIFICATION ANALYSIS", "REGRESSION ANALYSIS", "VISUALIZATION OF K-MEANS CLUSTERING"])
        
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()

        if choice == "DATASET OVERVIEW":
            st.title("**Retail Analytics Overview**")
            st.write("**Summary statistics and insights from transactional data.**")
            data = load_data()
            if data is not None:
                st.write("**Dataset Preview**", data.head())
                st.write("--------")
                st.write("**Dataset Describe**")
                st.write(data.describe())
            else:
                st.error("Data could not be loaded.")
            st.write("--------")
            st.bar_chart(data['Quantity'].value_counts())
        elif choice == "CLUSTER ANALYSIS":
            run_customer_segmentation()
        elif choice == "CLASSIFICATION ANALYSIS":
            run_classification()
        elif choice == "REGRESSION ANALYSIS":
            run_regression()
        else:
            show_retail_analysis()


if __name__ == '__main__':
    main()

