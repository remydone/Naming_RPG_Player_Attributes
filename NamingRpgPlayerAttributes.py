# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:08:39 2023

@author: Remy
"""

# Imports

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from gensim.models.keyedvectors import KeyedVectors


# Load Word2Vec model
def LoadModel(model_path='./GoogleNews-vectors-negative300.bin.gz'):
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return w2v_model


num_dims = 300


# Manually curated list of RPG player attributes

dnd5e = ["Strength", "Constitution", "Dexterity", "Inteligence",
         "Wisdom", "Charisma"]
traveller = ["Strength", "Dexterity", "Endurance", "Intelligence",
             "Education", "Social"]
cortex = ["Agility", "Alertness", "Intelligence", "Strength",
          "Vitality", "Willpower"]
tri = ["Body", "Mind", "Soul"]
storyteller_A = ["Physical", "Mental", "Social", "Power",
                 "Finesse", "Resilience"]
storyteller_B = ["Intelligence", "Strength", "Presence", "Wits", "Dexterity",
                 "Manipulation", "Resolve", "Stamina", "Composure"]
fallout = ["Strength", "Perception", "Endurance", "Charisma",
           "Intelligence", "Agility", "Luck"]
grups = ["Strength", "Dexterity", "Intelligence", "Health"]
bitd = ["Insight", "Prowess", "Resolve"]
bitd_sub = ["Hunt", "Study", "Survey", "Tinker", "finesse", "prowl",
            "skirmish", "wreck", "Attune", "Command", "Consort", "Sway"]

attributes = dnd5e+traveller+cortex+storyteller_A+storyteller_B+fallout+grups
attributes = [attribute.lower() for attribute in attributes]


def Main(w2v_model, attributes):
    # Attrubute vector loadings
    vectors = []
    for word in attributes:
        try:
            vec = w2v_model[word]
            vectors.append(vec)
        except KeyError:
            vectors.append([0]*num_dims)
    df = pd.DataFrame(vectors)
    df.columns = ['dim_%i' % i for i in range(df.shape[1])]
    df["attributes"] = attributes
    df_data = df[df.columns[:-1]]

    # Scale Data
    X = df_data.to_numpy()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # PCA
    new_num_dims = 100
    pca = PCA(n_components=new_num_dims)
    X_transformed = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data=X_transformed, index=df_data.index)
    df_pca.columns = ["dim_%i" % i for i in range(df_pca.shape[1])]
    df_pca["attributes"] = attributes
    df_pca_data = df_pca[df_pca.columns[:-1]]

    # kPCA
    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True)
    X_transformed = kpca.fit_transform(X_scaled)
    df_kpca = pd.DataFrame(data=X_transformed, index=df_data.index)
    df_kpca.columns = ["dim_%i" % i for i in range(df_kpca.shape[1])]
    df_kpca["attributes"] = attributes
    df_kpca_data = df_kpca[df_kpca.columns[:-1]]

    # K-means
    dist_list = []
    n_clusters = range(2, 12)
    for k in n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=0, init="k-means++")
        k_means_labels = kmeans.fit_predict(df_pca_data)

        # Find intra-cluster sum squared differences
        x_true = df_pca_data.to_numpy()
        x_pred = kmeans.cluster_centers_[k_means_labels, :]
        dist_list.append(((x_true-x_pred)**2).sum())

        # Print optimal attribute names
        df_pca["Cluster_k=%i" % k] = k_means_labels
        optimal_attributes = []
        for cluster in range(k):
            cluster_data = df_pca["Cluster_k=%i" % k] == cluster
            words_in_cluster = df_pca[cluster_data]["attributes"]
            # words_not_in_cluster = df_kpca[~cluster_data]["attributes"]
            name, smlrty = w2v_model.most_similar(positive=words_in_cluster)[0]
            optimal_attributes.append(name)
        print("For k = " + str(k) + ", : " + str(optimal_attributes))

    # Plot K-means results
    plt.plot(n_clusters, dist_list)
    plt.title('Sum squared differences vs. Number of clusters', fontsize=20)

    # ToDo: t-SNE graph of clusterizations
    # ToDo: Deal with sub-attributes differently
