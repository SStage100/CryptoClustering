# CryptoClustering
Module 11 Challenge
Welcome to my project on clustering cryptocurrencies using Principal Component Analysis (PCA) and K-Means! This project was a fantastic learning experience, and I hope you find it interesting and useful.

Project Overview
In this project, I aimed to group cryptocurrencies based on their price changes over various time periods. By reducing the dimensions of the data using PCA and then applying K-Means clustering, I was able to identify distinct clusters of cryptocurrencies.

Steps and Code Explanation
1. Load and Preprocess the Data
First, I loaded the cryptocurrency market data from a CSV file and selected the relevant numerical features. I then standardized the data to ensure all features contribute equally to the analysis.

A. import pandas as pd from sklearn.preprocessing import StandardScaler

B. # Load the data
crypto_df = pd.read_csv("crypto_market_data.csv", index_col="coin_id")

C. # Select numerical features and standardize the data
numerical_features = crypto_df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_features)

D. # Create a DataFrame with the scaled data
scaled_df = pd.DataFrame(scaled_data, index=crypto_df.index, columns=numerical_features.columns)

2. Principal Component Analysis (PCA)
Next, I used PCA to reduce the dimensionality of the data to three principal components. This step helps simplify the data while retaining most of the variance.

A. from sklearn.decomposition import PCA

B. # Apply PCA
pca_model = PCA(n_components=3)
pca_data = pca_model.fit_transform(scaled_df)

C. # Create a DataFrame with PCA results
pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2', 'PCA3'], index=scaled_df.index)

3. Determine the Optimal Number of Clusters
To find the best number of clusters, I used the Elbow method. I created a range of K values and calculated the inertia for each, then plotted the results to visually identify the "elbow" point.

A. from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

B. # Determine the optimal number of clusters using the Elbow method
k_values = list(range(1, 12))
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_df)
    inertia_values.append(kmeans.inertia_)

C. # Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.grid(True)
plt.show()

4. K-Means Clustering
Using the optimal number of clusters identified in the previous step, I applied K-Means clustering to the PCA data.

A. # Initialize and fit the K-Means model
best_k = 4  # This value is determined from the Elbow curve
kmeans_model = KMeans(n_clusters=best_k, random_state=42)
kmeans_model.fit(pca_df)

B. # Predict the clusters and add them to the DataFrame
pca_df['Cluster'] = kmeans_model.predict(pca_df)

5. Visualize the Clusters
To visualize the clusters, I created a scatter plot using hvPlot, which makes it easy to see how the cryptocurrencies are grouped based on the PCA components.

A. import hvplot.pandas

B. # Create a scatter plot using hvPlot
scatter_plot = pca_df.hvplot.scatter(
    x='PCA1',
    y='PCA2',
    c='Cluster',
    colormap='rainbow',
    title='Cryptocurrency Clusters (PCA Data)',
    hover_cols=['PCA3', 'Cluster']
)
scatter_plot

6. Analyze Feature Weights
Finally, I examined the weights of each feature on the principal components to understand which features had the strongest influence on each component.

A. # Get the principal component weights
pca_weights = pca_model.components_

B. # Create a DataFrame with the weights
pca_weights_df = pd.DataFrame(pca_weights.T, index=numerical_features.columns, columns=['PCA1', 'PCA2', 'PCA3'])

C. # Display the weights
print(pca_weights_df)

Key Findings
1. PCA1: price_change_percentage_200d and price_change_percentage_1y had the strongest positive influence, while price_change_percentage_24h had the strongest negative influence.
2. PCA2: price_change_percentage_30d and price_change_percentage_14d had the strongest positive influence, while price_change_percentage_1y had the strongest negative influence.
3. PCA3: price_change_percentage_7d had the strongest positive influence, while price_change_percentage_60d had the strongest negative influence.

Conclusion
This project helped me understand how to apply PCA and K-Means clustering to real-world data. I learned how to preprocess data, reduce dimensionality, find optimal clusters, and visualize the results. I hope this README provides a clear overview of my project and helps you understand the steps involved.

Feel free to explore the code and data further. Happy clustering!













