Key Features:
Data Generation & Exploration: Creates realistic customer data and provides thorough EDA

Data Preprocessing: StandardScaler for normalization

Dimensionality Reduction: PCA and t-SNE for visualization

Multiple Clustering Methods: K-Means and DBSCAN for comparison

Optimal Cluster Selection: Elbow method and silhouette scores

Comprehensive Visualization: Multiple plots including radar charts

Segment Interpretation: Automatic naming and profiling of customer segments

Actionable Insights: Marketing recommendations for each segment

Expected Output Segments:
High-Value Customers: High income, high spending

Budget Shoppers: Lower income, price-sensitive

Frequent Buyers: Regular purchasers regardless of spend

Young Spenders: Younger demographic with good spending

Standard Customers: Average across all metrics

Usage Tips:
For your own data: Replace the generate_customer_data() function with your actual dataset

Parameter Tuning: Adjust optimal_k, DBSCAN eps and min_samples based on your data

Segment Naming: Modify the interpretation logic to match your business context

Additional Features: Add more customer attributes for richer segmentation