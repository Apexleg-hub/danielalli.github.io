import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Generate sample customer spending data (in a real scenario, you'd load your own data)
def generate_customer_data(n_samples=300):
    np.random.seed(42)
    
    # Create different customer segments
    data = {
        'age': np.concatenate([
            np.random.normal(25, 3, n_samples//3),      # Young customers
            np.random.normal(45, 5, n_samples//3),      # Middle-aged
            np.random.normal(65, 4, n_samples//3)       # Senior customers
        ]),
        'annual_income': np.concatenate([
            np.random.normal(30000, 5000, n_samples//3),    # Budget shoppers
            np.random.normal(60000, 8000, n_samples//3),    # Middle-income
            np.random.normal(120000, 15000, n_samples//3)   # High-income
        ]),
        'spending_score': np.concatenate([
            np.random.normal(30, 10, n_samples//3),     # Low spenders
            np.random.normal(60, 15, n_samples//3),     # Moderate spenders
            np.random.normal(85, 8, n_samples//3)       # High spenders
        ]),
        'purchase_frequency': np.concatenate([
            np.random.poisson(2, n_samples//3),         # Infrequent buyers
            np.random.poisson(8, n_samples//3),         # Regular buyers
            np.random.poisson(15, n_samples//3)         # Frequent buyers
        ]),
        'avg_transaction_value': np.concatenate([
            np.random.normal(25, 8, n_samples//3),      # Small transactions
            np.random.normal(75, 20, n_samples//3),     # Medium transactions
            np.random.normal(200, 50, n_samples//3)     # Large transactions
        ])
    }
    
    df = pd.DataFrame(data)
    return df

# Load and explore data
print("Loading customer data...")
df = generate_customer_data()
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

# Visualize the original data distribution
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')

plt.subplot(2, 3, 2)
sns.histplot(df['annual_income'], kde=True)
plt.title('Annual Income Distribution')

plt.subplot(2, 3, 3)
sns.histplot(df['spending_score'], kde=True)
plt.title('Spending Score Distribution')

plt.subplot(2, 3, 4)
sns.histplot(df['purchase_frequency'], kde=True)
plt.title('Purchase Frequency Distribution')

plt.subplot(2, 3, 5)
sns.histplot(df['avg_transaction_value'], kde=True)
plt.title('Average Transaction Value Distribution')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Step 1: Data Normalization
print("Step 1: Data Normalization")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("Data normalized successfully!")

# Step 2: PCA for Dimensionality Reduction
print("\nStep 2: Principal Component Analysis")
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance Ratio')
plt.grid(True)
plt.show()

print(f"Variance explained by first 2 components: {pca.explained_variance_ratio_[:2].sum():.3f}")
print(f"Variance explained by first 3 components: {pca.explained_variance_ratio_[:3].sum():.3f}")

# Use 2 components for visualization
pca_2d = PCA(n_components=2)
pca_data_2d = pca_2d.fit_transform(scaled_data)

# Step 3: Determine Optimal Number of Clusters for K-Means
print("\nStep 3: Finding Optimal Number of Clusters")

# Method 1: Elbow Method
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)

# Method 2: Silhouette Score
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', color='red')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')
plt.grid(True)

plt.tight_layout()
plt.show()

# Choose optimal k (you can modify this based on the plots)
optimal_k = 4
print(f"Selected number of clusters: {optimal_k}")

# Step 4: Apply K-Means Clustering
print("\nStep 4: Applying K-Means Clustering")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to original data
df['kmeans_cluster'] = kmeans_labels
scaled_df['kmeans_cluster'] = kmeans_labels

# Step 5: Apply DBSCAN Clustering (for comparison)
print("\nStep 5: Applying DBSCAN Clustering")
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data)

df['dbscan_cluster'] = dbscan_labels
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN found {n_dbscan_clusters} clusters with {n_noise} noise points")

# Step 6: Visualize Clusters
print("\nStep 6: Visualizing Clusters")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. K-Means Clusters in PCA space
plt.subplot(2, 3, 1)
scatter = plt.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-Means Clusters (PCA-reduced)')
plt.colorbar(scatter)

# 2. DBSCAN Clusters in PCA space
plt.subplot(2, 3, 2)
scatter = plt.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('DBSCAN Clusters (PCA-reduced)')
plt.colorbar(scatter)

# 3. t-SNE visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_data = tsne.fit_transform(scaled_data)

plt.subplot(2, 3, 3)
scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('K-Means Clusters (t-SNE)')
plt.colorbar(scatter)

# 4. Original feature space visualization (Income vs Spending)
plt.subplot(2, 3, 4)
scatter = plt.scatter(df['annual_income'], df['spending_score'], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters: Income vs Spending')
plt.colorbar(scatter)

# 5. Age vs Purchase Frequency
plt.subplot(2, 3, 5)
scatter = plt.scatter(df['age'], df['purchase_frequency'], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Purchase Frequency')
plt.title('Clusters: Age vs Purchase Frequency')
plt.colorbar(scatter)

# 6. Cluster sizes
plt.subplot(2, 3, 6)
cluster_sizes = df['kmeans_cluster'].value_counts().sort_index()
plt.bar(cluster_sizes.index, cluster_sizes.values, color=plt.cm.viridis(np.linspace(0, 1, optimal_k)))
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Sizes')
for i, v in enumerate(cluster_sizes.values):
    plt.text(i, v + 1, str(v), ha='center')

plt.tight_layout()
plt.show()

# Step 7: Interpret Segments
print("\nStep 7: Interpreting Customer Segments")

# Analyze cluster characteristics
cluster_profiles = df.groupby('kmeans_cluster').agg({
    'age': 'mean',
    'annual_income': 'mean',
    'spending_score': 'mean',
    'purchase_frequency': 'mean',
    'avg_transaction_value': 'mean'
}).round(2)

print("\nCluster Profiles (Means):")
print(cluster_profiles)

# Create detailed cluster analysis
plt.figure(figsize=(15, 10))

# Radar chart function
def create_radar_chart(cluster_data, cluster_num, ax):
    categories = list(cluster_data.keys())[1:]  # Exclude cluster number
    values = list(cluster_data.values())[1:]
    
    # Normalize values for radar chart (0-1 scale)
    normalized_values = [(x - min(values)) / (max(values) - min(values)) for x in values]
    
    # Complete the circle
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    normalized_values += normalized_values[:1]
    
    ax.plot(angles, normalized_values, 'o-', linewidth=2, label=f'Cluster {cluster_num}')
    ax.fill(angles, normalized_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)

# Create radar chart for each cluster
fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

for i, cluster_num in enumerate(cluster_profiles.index[:4]):  # Show first 4 clusters
    cluster_data = cluster_profiles.loc[cluster_num]
    create_radar_chart(cluster_data, cluster_num, axes[i])
    axes[i].set_title(f'Cluster {cluster_num} Profile', size=14, weight='bold')

plt.tight_layout()
plt.show()

# Interpret each cluster
segment_descriptions = {}
for cluster in cluster_profiles.index:
    profile = cluster_profiles.loc[cluster]
    
    # Determine segment characteristics
    age_desc = "Young" if profile['age'] < 35 else "Middle-aged" if profile['age'] < 55 else "Senior"
    income_desc = "Low" if profile['annual_income'] < 45000 else "Medium" if profile['annual_income'] < 80000 else "High"
    spending_desc = "Low" if profile['spending_score'] < 40 else "Medium" if profile['spending_score'] < 70 else "High"
    
    # Create segment name and description
    if profile['annual_income'] > 80000 and profile['spending_score'] > 70:
        segment_name = "High-Value Customers"
        description = "Affluent customers with high spending power and frequent purchases"
    elif profile['annual_income'] < 45000 and profile['spending_score'] < 40:
        segment_name = "Budget Shoppers"
        description = "Price-sensitive customers with lower spending"
    elif profile['purchase_frequency'] > 10:
        segment_name = "Frequent Buyers"
        description = "Loyal customers who shop regularly"
    elif profile['age'] < 35 and profile['spending_score'] > 60:
        segment_name = "Young Spenders"
        description = "Young customers with good spending habits"
    else:
        segment_name = "Standard Customers"
        description = "Average customers with moderate spending patterns"
    
    segment_descriptions[cluster] = {
        'name': segment_name,
        'description': description,
        'profile': profile
    }

print("\n" + "="*50)
print("CUSTOMER SEGMENT INTERPRETATION")
print("="*50)

for cluster, info in segment_descriptions.items():
    print(f"\nCluster {cluster}: {info['name']}")
    print(f"Description: {info['description']}")
    print("Profile Summary:")
    print(f"  - Average Age: {info['profile']['age']:.1f} years")
    print(f"  - Average Income: ${info['profile']['annual_income']:,.0f}")
    print(f"  - Spending Score: {info['profile']['spending_score']:.1f}/100")
    print(f"  - Purchase Frequency: {info['profile']['purchase_frequency']:.1f} times/month")
    print(f"  - Avg Transaction: ${info['profile']['avg_transaction_value']:.2f}")

# Final comparison between clustering methods
print("\n" + "="*50)
print("CLUSTERING METHOD COMPARISON")
print("="*50)

kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
dbscan_silhouette = silhouette_score(scaled_data, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0

print(f"K-Means Results:")
print(f"  - Number of clusters: {optimal_k}")
print(f"  - Silhouette Score: {kmeans_silhouette:.3f}")
print(f"  - Cluster sizes: {dict(df['kmeans_cluster'].value_counts().sort_index())}")

print(f"\nDBSCAN Results:")
print(f"  - Number of clusters: {n_dbscan_clusters}")
print(f"  - Noise points: {n_noise}")
print(f"  - Silhouette Score: {dbscan_silhouette:.3f}")

# Recommendations based on segments
print("\n" + "="*50)
print("MARKETING RECOMMENDATIONS")
print("="*50)

for cluster, info in segment_descriptions.items():
    print(f"\n{info['name']} (Cluster {cluster}):")
    if "High-Value" in info['name']:
        print("  → Strategy: Premium loyalty programs, exclusive offers, personal shopping assistance")
    elif "Budget" in info['name']:
        print("  → Strategy: Discount campaigns, value bundles, price-sensitive marketing")
    elif "Frequent" in info['name']:
        print("  → Strategy: Loyalty rewards, early access to sales, referral programs")
    elif "Young" in info['name']:
        print("  → Strategy: Social media marketing, trend-focused products, flexible payment options")
    else:
        print("  → Strategy: General marketing campaigns, seasonal promotions, email newsletters")

print(f"\nTotal customers analyzed: {len(df)}")
print("Segmentation complete! Use these insights to tailor your marketing strategies.")