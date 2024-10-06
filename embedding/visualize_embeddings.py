
import pandas as pd
from langchain.vectorstores import Chroma
#pip install -U langchain-chroma
#from langchain_chroma import Chroma

def print_collection(collection):

    for i in range(0,len(collection)):
        doc_element = collection["documents"][i]
        meta_element = collection["metadatas"][i]
        embeddings_vector = collection["embeddings"][i]
        print("Document: ", doc_element)
        print("Metadata: ", meta_element)
        print("Embeddings: ", embeddings_vector)

train_embedding_file = "chromadb_train_embeddings.db"
test_embedding_file = "chromadb_test_embeddings.db"
special_data_embedding_file = "chromadb_ori_pqaa_embeddings.db"
pqau_embedding_file = "chromadb_ori_pqau_embeddings.db"


      
## load embeddings from train collection
train_vectordb = Chroma(persist_directory=train_embedding_file, collection_metadata={"hnsw:space": "cosine"})
train_vectordb_collection =train_vectordb._collection.get(include=["embeddings", "documents", "metadatas"])


## load embeddings from test collection
test_vectordb = Chroma(persist_directory=test_embedding_file, collection_metadata={"hnsw:space": "cosine"})
test_vectordb_collection =test_vectordb._collection.get(include=["embeddings", "documents", "metadatas"])

## load embeddings from special collection
special_vectordb = Chroma(persist_directory=special_data_embedding_file, collection_metadata={"hnsw:space": "cosine"})
special_vectordb_collection =special_vectordb._collection.get(include=["embeddings", "documents", "metadatas"])

## load embeddings from special collection
pqau_vectordb = Chroma(persist_directory=pqau_embedding_file, collection_metadata={"hnsw:space": "cosine"})
pqau_vectordb_collection =pqau_vectordb._collection.get(include=["embeddings", "documents", "metadatas"])

print("Length of train collection: ", len(train_vectordb_collection["documents"]))
print("Length of test collection: ", len(test_vectordb_collection["documents"]))
print("Length of special collection: ", len(special_vectordb_collection["documents"]))

train_embeddings_df = pd.DataFrame([
    train_vectordb_collection["embeddings"][i] for i in range(0,len(train_vectordb_collection["embeddings"]))
])

test_embeddings_df = pd.DataFrame([
    test_vectordb_collection["embeddings"][i] for i in range(0,len(test_vectordb_collection["embeddings"]))
])

specialdata_embeddings_df = pd.DataFrame([
    special_vectordb_collection["embeddings"][i] for i in range(0,len(special_vectordb_collection["embeddings"][:20000]))
])

pqau_embeddings_df = pd.DataFrame([
    pqau_vectordb_collection["embeddings"][i] for i in range(0,len(pqau_vectordb_collection["embeddings"]))
])



train_embeddings_df["source"] = "train"
test_embeddings_df["source"] = "test"
specialdata_embeddings_df["source"] = "special"
pqau_embeddings_df["source"] = "pqau"
#print(train_embeddings_df.head())
#print(test_embeddings_df.head()) 
#print(specialdata_embeddings_df.head()) 

#combined_embeddings_df = pd.concat([ test_embeddings_df, train_embeddings_df, specialdata_embeddings_df])
combined_embeddings_df = pd.concat([ test_embeddings_df, train_embeddings_df, pqau_embeddings_df])


## reduce dimensions to 3D
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

combined_embeddings_3d_df = pca.fit_transform(combined_embeddings_df.drop(columns=["source"]))

print(combined_embeddings_3d_df.shape)
print(combined_embeddings_3d_df)


## cluster embeddings
SHOW_CLUSTER_CENTERS = False
if SHOW_CLUSTER_CENTERS:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(combined_embeddings_3d_df)
    #print(kmeans.labels_)
    clusters_df = pd.DataFrame(kmeans.cluster_centers_)
    #print(kmeans.cluster_centers_)
    #print(clusters_df)
    #print(combined_embeddings_3d_df)
    #combined_embeddings_df["cluster"] = kmeans.labels_
    #print(combined_embeddings_df.head())


## visualize embeddings as 3D scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


x = combined_embeddings_3d_df[:,0]
y = combined_embeddings_3d_df[:,1]
z = combined_embeddings_3d_df[:,2]
#colors_column = combined_embeddings_df["source"].map({"test": "red", "train": "blue", "special": "green"})
colors_column = combined_embeddings_df["source"].map({"test": "red", "train": "blue", "pqau": "purple"})

if SHOW_CLUSTER_CENTERS:
    x = clusters_df[0]
    y = clusters_df[1]
    z = clusters_df[2]
    colors_column = "c"



ax.scatter(x, y, z, c=colors_column, marker='o', label=combined_embeddings_df["source"])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter plot of embeddings')

## create legend for colors
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Orial-test')
blue_patch = mpatches.Patch(color='blue', label='Orial-train')
#green_patch = mpatches.Patch(color='green', label='PQAA')
purple_patch = mpatches.Patch(color='purple', label='PQAU')

#ax.legend(handles=[red_patch,blue_patch,green_patch])
ax.legend(handles=[red_patch,blue_patch,purple_patch])



plt.show()
