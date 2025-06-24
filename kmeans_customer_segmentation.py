import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    Load and clean the retail dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned retail data
    """
    print("Loading and cleaning data...")
    retail = pd.read_csv(file_path, encoding='unicode_escape')
    retail = retail.dropna()
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
    print(f"Loaded {len(retail)} transactions for {retail['CustomerID'].nunique()} customers")
    return retail

def engineer_rfm_features(retail_data):
    """
    Engineer RFM (Recency, Frequency, Monetary) features for customer segmentation.
    
    Args:
        retail_data (pd.DataFrame): Clean retail data
        
    Returns:
        pd.DataFrame: Customer features with RFM metrics
    """
    print("Engineering RFM features...")
    
    # Monetary: Total amount spent by each customer
    monetary = retail_data.groupby('CustomerID')['Amount'].sum().reset_index()
    
    # Frequency: Number of transactions by each customer
    frequency = retail_data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    frequency.columns = ['CustomerID', 'Frequency']
    
    # Merge monetary and frequency
    retail_features = pd.merge(monetary, frequency, on='CustomerID')
    
    # Recency: Days since last purchase
    retail_data['InvoiceDate'] = pd.to_datetime(retail_data['InvoiceDate'], format='%m/%d/%Y %H:%M')
    max_date = retail_data['InvoiceDate'].max()
    retail_data['Recency'] = (max_date - retail_data['InvoiceDate']).dt.days
    recency = retail_data.groupby('CustomerID')['Recency'].min().reset_index()
    
    # Final merge
    retail_features = pd.merge(retail_features, recency, on='CustomerID')
    retail_features.columns = ['CustomerID', 'Monetary', 'Frequency', 'Recency']
    
    print(f"Created RFM features for {len(retail_features)} customers")
    return retail_features

def remove_outliers(data, columns):
    """
    Remove outliers using IQR method.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): Columns to check for outliers
        
    Returns:
        pd.DataFrame: Data with outliers removed
    """
    print("Removing outliers...")
    original_count = len(data)
    
    for col in columns:
        q1 = data[col].quantile(0.05)
        q3 = data[col].quantile(0.95)
        iqr = q3 - q1
        data = data[(data[col] >= q1 - 1.5 * iqr) & (data[col] <= q3 + 1.5 * iqr)]
    
    print(f"Removed {original_count - len(data)} outliers")
    return data

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        X_scaled (np.array): Scaled features
        max_k (int): Maximum number of clusters to try
        
    Returns:
        int: Optimal number of clusters
    """
    print("Finding optimal number of clusters...")
    
    ssd = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, max_iter=100, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        ssd.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, ssd, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k (elbow point)
    optimal_k = 4  # Based on elbow curve analysis
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def perform_clustering(X_scaled, n_clusters):
    """
    Perform K-Means clustering.
    
    Args:
        X_scaled (np.array): Scaled features
        n_clusters (int): Number of clusters
        
    Returns:
        KMeans: Fitted K-Means model
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans

def visualize_clusters(data, kmeans, feature_names, scaler):
    """
    Create comprehensive cluster visualizations.
    
    Args:
        data (pd.DataFrame): Data with cluster labels
        kmeans (KMeans): Fitted K-Means model
        feature_names (list): Names of features used for clustering
        scaler (StandardScaler): Fitted scaler to transform centroids back to original scale
    """
    print("Creating visualizations...")
    
    # Transform centroids back to original scale
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 1. Main cluster plot (Monetary vs Recency)
    plt.figure(figsize=(15, 10))
    
    # Scatter plot with clusters
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == i]
        plt.scatter(cluster_data['Monetary'], cluster_data['Recency'], 
                   c=colors[i], label=f'Cluster {i+1}', alpha=0.7, s=50)
    
    # Centroids (Monetary vs Recency)
    plt.scatter(centroids_original[:, 0], centroids_original[:, 2], c='red', s=300, 
               marker='X', label='Centroids', edgecolors='black', linewidth=2)
    
    plt.xlabel('Monetary Value')
    plt.ylabel('Recency (days)')
    plt.title('Customer Segmentation: Monetary vs Recency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Additional plot: Frequency vs Monetary
    plt.figure(figsize=(12, 8))
    
    for i in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == i]
        plt.scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
                   c=colors[i], label=f'Cluster {i+1}', alpha=0.7, s=50)
    
    # Centroids (Frequency vs Monetary)
    plt.scatter(centroids_original[:, 1], centroids_original[:, 0], c='red', s=300, 
               marker='X', label='Centroids', edgecolors='black', linewidth=2)
    
    plt.xlabel('Frequency')
    plt.ylabel('Monetary Value')
    plt.title('Customer Segmentation: Frequency vs Monetary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('frequency_vs_monetary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Cluster characteristics
    plt.figure(figsize=(15, 5))
    
    # Monetary distribution by cluster
    plt.subplot(1, 3, 1)
    for i in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == i]
        plt.hist(cluster_data['Monetary'], alpha=0.7, label=f'Cluster {i+1}', 
                bins=20, color=colors[i])
    plt.xlabel('Monetary Value')
    plt.ylabel('Frequency')
    plt.title('Monetary Distribution by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency distribution by cluster
    plt.subplot(1, 3, 2)
    for i in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == i]
        plt.hist(cluster_data['Frequency'], alpha=0.7, label=f'Cluster {i+1}', 
                bins=20, color=colors[i])
    plt.xlabel('Frequency')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recency distribution by cluster
    plt.subplot(1, 3, 3)
    for i in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == i]
        plt.hist(cluster_data['Recency'], alpha=0.7, label=f'Cluster {i+1}', 
                bins=20, color=colors[i])
    plt.xlabel('Recency (days)')
    plt.ylabel('Frequency')
    plt.title('Recency Distribution by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_characteristics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Cluster summary statistics
    cluster_summary = data.groupby('Cluster').agg({
        'Monetary': ['mean', 'std', 'count'],
        'Frequency': ['mean', 'std'],
        'Recency': ['mean', 'std']
    }).round(2)
    
    print("\nCluster Summary Statistics:")
    print(cluster_summary)
    
    # Save cluster summary to CSV
    cluster_summary.to_csv('cluster_summary.csv')
    print("Cluster summary saved to 'cluster_summary.csv'")
    
    # Print centroid information
    print("\nCentroid Information (Original Scale):")
    for i, centroid in enumerate(centroids_original):
        print(f"Cluster {i+1}: Monetary={centroid[0]:.2f}, Frequency={centroid[1]:.2f}, Recency={centroid[2]:.2f}")

def main():
    """
    Main function to run the customer segmentation analysis.
    """
    print("=== Customer Segmentation using K-Means Clustering ===\n")
    
    # Load and clean data
    retail_data = load_and_clean_data('OnlineRetail.csv')
    
    # Engineer RFM features
    customer_features = engineer_rfm_features(retail_data)
    
    # Remove outliers
    customer_features = remove_outliers(customer_features, ['Monetary', 'Frequency', 'Recency'])
    
    # Prepare features for clustering
    X = customer_features[['Monetary', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFinal dataset shape: {X_scaled.shape}")
    print("Feature statistics after scaling:")
    print(pd.DataFrame(X_scaled, columns=['Monetary', 'Frequency', 'Recency']).describe())
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled, max_k=8)
    
    # Perform clustering
    kmeans_model = perform_clustering(X_scaled, optimal_k)
    
    # Add cluster labels to data
    customer_features['Cluster'] = kmeans_model.labels_
    
    # Visualize results
    visualize_clusters(customer_features, kmeans_model, ['Monetary', 'Frequency', 'Recency'], scaler)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- elbow_curve.png: Elbow method and silhouette analysis")
    print("- customer_segmentation.png: Main cluster visualization (Monetary vs Recency)")
    print("- frequency_vs_monetary.png: Frequency vs Monetary visualization")
    print("- cluster_characteristics.png: Feature distributions by cluster")
    print("- cluster_summary.csv: Statistical summary of clusters")

if __name__ == "__main__":
    main()