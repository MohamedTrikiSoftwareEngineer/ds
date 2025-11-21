import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("Segmentation de Restaurants Zomato")

# Chargement du dataset
df = pd.read_csv('zomato.csv', encoding='latin1')

st.write(df.head())

# Exploration des données
st.write(df.describe())

# Sélection des variables
X = df[['Average Cost for two', 'Aggregate rating']]

# Distribution des données
st.write("Distribution des données")
fig0 = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(X['Average Cost for two'], bins=30, edgecolor='black')
plt.xlabel('Coût Moyen pour Deux')
plt.ylabel('Fréquence')
plt.title('Distribution du Coût')

plt.subplot(1, 2, 2)
plt.hist(X['Aggregate rating'], bins=20, edgecolor='black')
plt.xlabel('Note Moyenne')
plt.ylabel('Fréquence')
plt.title('Distribution des Notes')

st.pyplot(fig0)

# Visualisation des données
st.write("Visualisation des données")
fig1 = plt.figure()
plt.scatter(X['Average Cost for two'], X['Aggregate rating'], s=50)
plt.xlabel('Coût Moyen pour Deux')
plt.ylabel('Note Moyenne')
plt.title('Restaurants')
st.pyplot(fig1)

# Méthode du coude
st.write("Méthode du coude")
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia = inertia + [km.inertia_]

fig2 = plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour choisir k')
plt.grid(True)
st.pyplot(fig2)

# Méthode de silhouette
st.write("Méthode de score de silhouette")
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels_temp = km.fit_predict(X)
    score = silhouette_score(X, labels_temp)
    silhouette_scores = silhouette_scores + [score]
    st.text(f"Pour k={k}, le score de silhouette moyen est {score:.3f}")

fig3 = plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Score de silhouette moyen pour chaque valeur de k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score de silhouette moyen")
plt.grid(True)
st.pyplot(fig3)

# K optimal
k_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
st.write(f"**K optimal sélectionné: {k_optimal}**")

# Application de K-means
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualisation des clusters
st.write(f"Segmentation des restaurants en {k_optimal} clusters")
fig4 = plt.figure(figsize=(8, 6))
sns.scatterplot(x=X['Average Cost for two'], y=X['Aggregate rating'],
                hue=labels, palette='Set1', s=60)
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.6, marker='X', label='Centroïdes')
plt.xlabel('Coût Moyen pour Deux')
plt.ylabel('Note Moyenne')
plt.title(f'Segmentation par K-means (K={k_optimal})')
plt.legend()
st.pyplot(fig4)

# Répartition des clusters
st.write("Répartition des points par cluster")
X_clustered = X.copy()
X_clustered['Cluster'] = labels

cluster_counts = []
for i in range(k_optimal):
    cluster_count = (labels == i).sum()
    cluster_counts = cluster_counts + [cluster_count]
    pourcentage = (cluster_count / len(labels)) * 100
    st.text(f"Cluster {i}: {cluster_count} restaurants ({pourcentage:.1f}%)")

# Graphique à barres corrigé
fig5 = plt.figure(figsize=(8, 5))
plt.bar(range(k_optimal), cluster_counts, edgecolor='black', color='steelblue')
plt.xlabel('Cluster')
plt.ylabel('Nombre de restaurants')
plt.title(f'Répartition en {k_optimal} clusters')
plt.xticks(range(k_optimal))  # Afficher tous les numéros de clusters
st.pyplot(fig5)
