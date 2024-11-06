from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Function to create the adjacency matrix
def create_adjacent_matrix(df):
    nodes = len(df)
    adjacency_matrix = np.zeros((nodes, nodes), dtype=int)

    for i in range(nodes):
        adjacency_matrix[i] = df.geometry.intersects(df.geometry[i]).astype(int)
        adjacency_matrix[i, i] = 0  # Remove self-adjacency

    return adjacency_matrix

# Perform clustering to assign labels without modifying the adjacency matrix
def label_clusters(df, adjacency_matrix, n_clusters=5):
    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='precomputed', linkage='average'
    )
    # Invert the adjacency matrix for clustering (to treat it as similarity)
    labels = clustering_model.fit_predict(1 - adjacency_matrix)
    df['cluster'] = labels
    return df

# Sort comarcas and modify the adjacency matrix based on cluster order
def sort_and_modify_adjacency(df, adjacency_matrix):
    # Sort by cluster label and then by original index
    sorted_df = df.sort_values(by=['cluster']).reset_index(drop=True)
    
    # Create an ordered index mapping
    ordered_indices = sorted_df.index
    
    # Reorder the adjacency matrix
    ordered_adjacency_matrix = adjacency_matrix[ordered_indices][:, ordered_indices]
    
    return sorted_df, ordered_adjacency_matrix

# Main function to get clusters and order comarcas within the clusters
def sorted_adjacency_matrix_with_clusters(df, n_clusters=5):
    adjacency_matrix = create_adjacent_matrix(df)
    clustered_df = label_clusters(df, adjacency_matrix, n_clusters=n_clusters)
    ordered_df, ordered_adjacency_matrix = sort_and_modify_adjacency(clustered_df, adjacency_matrix)
    return ordered_df, ordered_adjacency_matrix

def get_adjacent_comarcas(df, adjacency_matrix):
    adjacent_list = {}

    for i in range(len(adjacency_matrix)):
        comarca_name = df.iloc[i]['NOMCOMAR']
        # Find indices of adjacent comarcas where adjacency_matrix[i][j] == 1
        adjacent_comarcas = df.loc[adjacency_matrix[i] == 1, 'NOMCOMAR'].tolist()
        adjacent_list[comarca_name] = adjacent_comarcas

    return adjacent_list