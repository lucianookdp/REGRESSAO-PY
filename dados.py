import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Carregar os dados
file_path = 'spotify-2023.csv'
spotify_data = pd.read_csv(file_path, encoding='latin1')

# Escolher as colunas para usar na análise
numerical_columns = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                     'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
spotify_numeric = spotify_data[numerical_columns]

# Passo 1: Tratamento de Outliers usando IQR
Q1 = spotify_numeric.quantile(0.25)
Q3 = spotify_numeric.quantile(0.75)
IQR = Q3 - Q1

# Definindo os limites inferior e superior para detecção de outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrando os dados para remover outliers
spotify_no_outliers = spotify_numeric[~((spotify_numeric < lower_bound) | (spotify_numeric > upper_bound)).any(axis=1)]

# Passo 2: Padronizar os dados originais e os dados sem outliers
scaler = StandardScaler()
spotify_scaled = scaler.fit_transform(spotify_numeric)  # Dados originais padronizados
spotify_scaled_no_outliers = scaler.fit_transform(spotify_no_outliers)  # Dados sem outliers padronizados

# Aplicar K-Means nos dados sem outliers
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(spotify_scaled_no_outliers)

# Aplicar Hierarchical Clustering e DBSCAN nos dados originais padronizados
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(spotify_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(spotify_scaled)

# Calcular as métricas de avaliação
silhouette_kmeans = silhouette_score(spotify_scaled_no_outliers, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(spotify_scaled_no_outliers, kmeans_labels)

silhouette_hierarchical = silhouette_score(spotify_scaled, hierarchical_labels)
davies_bouldin_hierarchical = davies_bouldin_score(spotify_scaled, hierarchical_labels)

if len(set(dbscan_labels)) > 1:  # Verificar se há mais de um cluster
    silhouette_dbscan = silhouette_score(spotify_scaled, dbscan_labels)
    davies_bouldin_dbscan = davies_bouldin_score(spotify_scaled, dbscan_labels)
else:
    silhouette_dbscan = None
    davies_bouldin_dbscan = None

# Mostrar as métricas
print("Métricas de Avaliação:")
print(f"K-Means (com tratamento de outliers): Silhouette = {silhouette_kmeans:.3f}, Davies-Bouldin = {davies_bouldin_kmeans:.3f}")
print(f"Hierarchical: Silhouette = {silhouette_hierarchical:.3f}, Davies-Bouldin = {davies_bouldin_hierarchical:.3f}")
print(f"DBSCAN: Silhouette = {silhouette_dbscan}, Davies-Bouldin = {davies_bouldin_dbscan}")

# Escolher o melhor algoritmo e calcular as estatísticas descritivas
spotify_data['Cluster'] = hierarchical_labels
cluster_stats = spotify_data.groupby('Cluster').agg({
    'bpm': ['mean', 'median', 'std'],
    'danceability_%': ['mean', 'median', 'std'],
    'valence_%': ['mean', 'median', 'std'],
    'energy_%': ['mean', 'median', 'std'],
    'acousticness_%': ['mean', 'median', 'std'],
    'instrumentalness_%': ['mean', 'median', 'std'],
    'liveness_%': ['mean', 'median', 'std'],
    'speechiness_%': ['mean', 'median', 'std']
})

# Organizando a tabela
cluster_stats = cluster_stats.round(2)
cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

print("\nEstatísticas Descritivas por Cluster:")
print(cluster_stats)
