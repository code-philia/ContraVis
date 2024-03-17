"""
Mapper: transfomrmation different representations
high d to high d
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
sys.path.append('..')
import torch
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from pynndescent import NNDescent
from sklearn.utils import check_random_state
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

RANDOM_STATE = 10
# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim=512, encoding_dim=256,hidden_dim=256):
        super(Autoencoder, self).__init__()
        
        # Encoder: tar to ref
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )
        
        # Decoder: ref to tar
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class TransformModelTrainer():
    def __init__(self,ref_data,tar_data,ref_provider,tar_provider,ref_epoch,tar_epoch,device,n_neighbors=15) -> None:
        self.ref_data = ref_data
        self.tar_data = tar_data
        # self.ref_proxy = ref_proxy
        # self.tar_proxy = tar_proxy
        self.n_neighbors = n_neighbors
        self.ref_provider = ref_provider
        self.tar_provider = tar_provider
        self.ref_epoch = ref_epoch
        self.tar_epoch= tar_epoch
        self.device = device
        self.model = Autoencoder().to(device)
        self.ref_data = self.ref_data.reshape(self.ref_data.shape[0],self.ref_data.shape[1])
        self.tar_data = self.tar_data.reshape(self.tar_data.shape[0],self.tar_data.shape[1])
        self.tar_pred = self.tar_provider.get_pred(self.tar_epoch, self.tar_data)
        self.ref_pred = self.ref_provider.get_pred(self.ref_epoch, self.ref_data)

    
    
    def init_aligned_clustering(self, ref_n_clusters=10, tar_n_clusters=10):
        """
            use clusering get aligned clusering centers
        """
        
        refClusterer = KMeans(n_clusters=ref_n_clusters, random_state=RANDOM_STATE)
        ref_cluster_labels = refClusterer.fit_predict(self.ref_data)
        ref_centers = refClusterer.cluster_centers_
        
        tarClusterer = KMeans(n_clusters=tar_n_clusters, random_state=RANDOM_STATE)
        tar_cluster_labels = tarClusterer.fit_predict(self.tar_data)
        tar_centers = tarClusterer.cluster_centers_
        
        ref_c_pred = self.ref_provider.get_pred(self.ref_epoch, ref_centers)
        tar_c_pred = self.tar_provider.get_pred(self.tar_epoch, tar_centers)
        
        cosine_sim_matrix = cosine_similarity(ref_c_pred, tar_c_pred)
        matches = np.argmax(cosine_sim_matrix, axis=1)
        tar_centers = tar_centers[matches]
        
        return ref_centers, tar_centers
    
    def consistency_score(self, distance_to_c_ref, distance_to_c_tar):
        """
            distance sort consistency
        """
        scores = []
        for ref_distances, tar_distances in zip(distance_to_c_ref, distance_to_c_tar):
            # sort
            ref_sorted_indices = np.argsort(ref_distances)[:3]
            tar_sorted_indices = np.argsort(tar_distances)[:3]
            # 检查最近距离是否不一致
            if ref_sorted_indices[0] != tar_sorted_indices[0]:
            # 如果最近的距离不一致，则给予显著的负得分
                scores.append(-1)
            else:
                # 否则，计算Spearman等级相关系数
                # Spearman corelation 
                coef, _ = spearmanr(ref_sorted_indices, tar_sorted_indices)
                scores.append(coef)
    
        return np.array(scores)

    def global_topology_similarity(self):
        """
            Calculates the global topology similarity by assessing the consistency of distances 
            between sample pairs and their respective cluster centers. This method aims to quantify 
            how similarly samples are positioned relative to the centers in both the reference and 
            target datasets, thus providing a measure of global alignment between the two datasets.
            
            The process involves the following steps:
                1. Initializing aligned cluster centers for both reference and target datasets using a predefined method.
                2. Computing the Euclidean distances of each sample to all centers in both datasets.
                3. Evaluating the consistency of these distances across the two datasets to derive similarity scores.

            Returns:
                np.array: Similarity scores indicating the consistency of sample-to-center distances across datasets.
                np.array: Distances from reference data samples to each of the n centers.
                np.array: Distances from target data samples to each of the n centers.
        """
        print("start calculating global topology similairty")
        self.ref_centers, self.tar_centers = self.init_aligned_clustering()
        diff_ref = self.ref_data[:, np.newaxis, :] - self.ref_centers[np.newaxis, :, :]
        distance_to_c_ref = np.sqrt(np.sum(diff_ref ** 2, axis=2))
        diff_tar = self.tar_data[:, np.newaxis, :] - self.tar_centers[np.newaxis, :, :]
        distance_to_c_tar = np.sqrt(np.sum(diff_tar ** 2, axis=2))
        scores = self.consistency_score(distance_to_c_ref, distance_to_c_tar)
        return scores, distance_to_c_ref, distance_to_c_tar
        
    def local_topology_similarity(self, k_neighbour=15):
        """
            Computes the local topology similarity between the reference and target datasets by examining the neighborhood overlap for each sample. This method evaluates the similarity in local structures of the datasets by counting the number of common nearest neighbors each sample has in both datasets.

            The process involves:
            1. Identifying the k nearest neighbors for each sample in both the reference and target datasets using a k-nearest neighbors (k-NN) algorithm.
            2. Comparing the nearest neighbors of each sample in the reference dataset with those in the target dataset to find common neighbors.
            3. Counting the number of common neighbors for each sample to quantify local topology similarity.

            Returns:
                np.array: An array containing the count of common nearest neighbors for each sample, representing the local topology similarity.
                np.array: Indices of the k nearest neighbors for each sample in the reference dataset.
                np.array: Indices of the k nearest neighbors for each sample in the target dataset.
        """
        print("start calculating local topology similairty")
        ref_nbrs = NearestNeighbors(n_neighbors=k_neighbour, algorithm='auto').fit(self.ref_data)
        _, ref_nn_indices = ref_nbrs.kneighbors(self.ref_data)
        tar_nbrs = NearestNeighbors(n_neighbors=k_neighbour, algorithm='auto').fit(self.tar_data)
        _, tar_nn_indices = tar_nbrs.kneighbors(self.tar_data)
        common_nn_counts = np.zeros(len(ref_nn_indices), dtype=int)
        for i in range(len(ref_nn_indices)):
            common_nn_counts[i] = np.intersect1d(ref_nn_indices[i], tar_nn_indices[i]).size
        return common_nn_counts, ref_nn_indices, tar_nn_indices
    
    def filter_(self, min_score=0, min_common_neibour=2):
        print("start filtering")
        unaligned_indicates = []
        scores, distance_to_c_ref, distance_to_c_tar = self.global_topology_similarity()
        common_nn_counts, self.ref_nn_indices, self.tar_nn_indices = self.local_topology_similarity()
        tar_pred = self.tar_pred.argmax(axis=1)
        ref_pred = self.ref_pred.argmax(axis=1)
        for i in range(len(tar_pred)):
            if not (scores[i] > min_score and common_nn_counts[i] >= min_common_neibour):
                unaligned_indicates.append(i)
        return unaligned_indicates
    
    def calculate_global_similarity(self, tar_i, ref_samples):
        """
        Calculate the global similarity between a target sample and a set of reference samples based on their distances to aligned centers.

        Parameters:
            tar_i (np.array): A single sample from the target dataset.
            ref_samples (np.array): A set of samples from the reference dataset.

        Returns:
            np.array: Similarity scores between tar_i and each sample in ref_samples.
        """
        # Calculate distances from tar_i to each aligned center
        diff_tar = tar_i - self.tar_centers
        distance_to_c_tar = np.sqrt(np.sum(diff_tar ** 2, axis=1))

        # Initialize a list to store similarity scores
        scores = []

        # Iterate over each reference sample
        for ref_sample in ref_samples:
            # Calculate distances from ref_sample to each aligned center
            diff_ref = ref_sample - self.ref_centers
            distance_to_c_ref = np.sqrt(np.sum(diff_ref ** 2, axis=1))

            # Sort distances
            ref_sorted_indices = np.argsort(distance_to_c_ref)
            tar_sorted_indices = np.argsort(distance_to_c_tar)

            # Check if the nearest distance is inconsistent
            if ref_sorted_indices[0] != tar_sorted_indices[0]:
                # If the nearest distance is inconsistent, assign a significant negative score
                scores.append(-1)
            else:
                # Otherwise, calculate the Spearman rank correlation coefficient
                coef, _ = spearmanr(ref_sorted_indices, tar_sorted_indices)
                scores.append(coef)

        return np.array(scores)
    
    def find_replacement_for_unaligned(self, tar_index):
        ref_indicates = self.ref_nn_indices[tar_index]
        ref_samples = self.ref_data[ref_indicates]
        best_score = -np.inf  # Initialize with a very low score
        best_ref_new_index = None
        global_scores = self.calculate_global_similarity(self.tar_data[tar_index], ref_samples)
        nn_indicates = self.tar_nn_indices[tar_index]
        ref_sample_nn = self.ref_nn_indices[ref_indicates]
        common_nn_counts = np.zeros(len(ref_sample_nn), dtype=int)
        for i in range(len(ref_sample_nn)):
            common_nn_counts[i] = np.intersect1d(ref_sample_nn, nn_indicates).size
        # Now best_ref_new_index points to the best replacement sample in ref_data
        valid_indices = np.where((common_nn_counts >= 2) & (global_scores > 0))[0]
        # 如果存在有效索引，选择全局相似度得分最高的参考样本
        if len(valid_indices) > 0:
            # 获取满足条件的最高全局相似度得分的索引
            best_match_index = valid_indices[np.argmax(global_scores[valid_indices])]
            return best_match_index, ref_samples[best_match_index]
        else:
            # 如果没有找到满足条件的样本，返回None或其他适当的值
            return None, None
  
    def random_sample(self, array, tar_pred, sample_size):
        """
        Randomly sample a subset from a numpy array.

        Args:
        - array (numpy.ndarray): The input array to sample from.
        - tar_pred (int): The target prediction to match.
        - sample_size (int): The size of the random sample.

        Returns:
        - numpy.ndarray: A subset of the input array.
        - numpy.ndarray: Indices of the sampled elements in the original array.
        """
        array_res = array.argmax(axis=1)

        # Find the indices where the condition matches
        matching_indices = np.where(array_res == tar_pred)[0]
        print("Number of matching elements:", len(matching_indices))

        # Randomly sample indices from the matching ones
        sampled_indices = np.random.choice(matching_indices, size=sample_size, replace=False)

        return array[sampled_indices], sampled_indices
    

    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    
    def cosine_distance(self, vec1, vec2):
        cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1 - cosine_similarity

    def align_ref_data(self,sample_size=100):
        diff_indicates = self.filter_()
        print("diff is", len(diff_indicates))
        tar_pred_res = self.tar_pred.argmax(axis=1)
        ref_pred = self.ref_pred
        need_Remove = []
        for i in diff_indicates:
            # Randomly sample from ref_pred
            # sampled_ref_pred, sampled_indices = self.random_sample(ref_pred, tar_pred_res[i], sample_size)
            new_ref_index,new_ref = self.find_replacement_for_unaligned(i)
            # Calculate the distance with the sampled ref points
            # distances = [self.cosine_distance(tar_pred[i], sampled_ref_pred[j]) for j in range(sample_size)]
            # # Find the most similar point
            # similar_ref_index = np.argmin(distances)
            # Get the original index in ref_data
            if new_ref is not None:
                self.ref_data[i] = new_ref
                need_Remove.append(i) ## no matched then remove
        return need_Remove
    
    def transformation_train(self,lambda_translation=0.5,num_epochs = 100,lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        need_Remove = self.align_ref_data()
        print("finished aligned",need_Remove)
        #TODO
        # self.tar_train = np.concatenate((self.tar_data, self.tar_proxy),axis=0)
        # self.ref_train = np.concatenate((self.ref_data, self.ref_proxy),axis=0)
        self.tar_train = self.tar_data
        self.ref_train = self.ref_data
        tar_tensor = torch.tensor(self.tar_train, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_train, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)
        # knn_overlap_loss = KNNOverlapLoss(k=15)

        # Train the autoencoder
        for epoch in range(num_epochs):
            self.model.train()

            optimizer.zero_grad()

            # Forward pass
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            # Compute the two losses
            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss
            # + lambda_knn * (knn_loss_encoder + knn_loss_decoder) 
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # To get the mapped version of tar
        tar_mapped = self.model.encoder(tar_tensor).cpu().detach().numpy()

        tar_data_shape = self.tar_data.shape[0]
        tar_data_mapped = tar_mapped[:tar_data_shape]
        tar_proxy_mapped = tar_mapped[tar_data_shape:]

        

        # To get ref back from the mapped tar
        ref_reconstructed = self.model.decoder(torch.tensor(tar_mapped).to(self.device)).cpu().detach().numpy()

        return self.model, tar_data_mapped, tar_proxy_mapped, ref_reconstructed
    

    

