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
    
class TransformationTrainer():
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
    
    def filter_diff(self):
        diff_indicates = []

        tar_pred = self.tar_pred.argmax(axis=1)
        ref_pred = self.ref_pred.argmax(axis=1)
        for i in range(len(tar_pred)):
            if tar_pred[i] != ref_pred[i]:
                diff_indicates.append(i)
        return diff_indicates
    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    
    def cosine_distance(self, vec1, vec2):
        cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1 - cosine_similarity

    def align_ref_data(self,sample_size=500):
        
        diff_indicates = self.filter_diff()
        print("diff is", len(diff_indicates))
        tar_pred = self.tar_pred
        tar_pred_res = self.tar_pred.argmax(axis=1)
        ref_pred = self.ref_pred

        # for i in diff_indicates:
        #     # Randomly sample from ref_pred
        #     sampled_ref_pred, sampled_indices = self.random_sample(ref_pred, tar_pred_res[i], sample_size)
        #     # calculate the distance
        #     # calculate the distance only with the sampled ref points
        #     distances = [self.cosine_distance(tar_pred[i], sampled_ref_pred[j]) for j in range(sample_size)]
        #     # find the most similar point
        #     similar_ref_index = np.argmin(distances)
        #     # replace
        #     self.ref_data[i] = self.ref_data[similar_ref_index]
        for i in diff_indicates:
            # Randomly sample from ref_pred
            sampled_ref_pred, sampled_indices = self.random_sample(ref_pred, tar_pred_res[i], sample_size)
            # Calculate the distance with the sampled ref points
            distances = [self.cosine_distance(tar_pred[i], sampled_ref_pred[j]) for j in range(sample_size)]
            # Find the most similar point
            similar_ref_index = np.argmin(distances)
            # Get the original index in ref_data
            original_ref_index = sampled_indices[similar_ref_index]
            # Replace
            self.ref_data[i] = self.ref_data[original_ref_index]
    

    
    def transformation_train(self,lambda_translation=0.5,num_epochs = 100,lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        self.align_ref_data()
        print("finished aligned")
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
    

    def compute_neighbors(self, data, n_neighbors=15):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(data) # 加1是因为第一个邻居是数据点自己
        _, indices = nbrs.kneighbors(data)
        return indices[:, 1:] # 去掉第一个邻居，因为它是数据点自己

    def similarity_loss(self, mapped_data, original_neighbors):
        mapped_neighbors = self.compute_neighbors(mapped_data.cpu().detach().numpy())

        # calculate the match score
        match_score = 0
        for m_neigh, o_neigh in zip(mapped_neighbors, original_neighbors):
            match_score += len(set(m_neigh) & set(o_neigh))
        
        # most match is 15 
        max_match_score = 15 * len(mapped_data)  # Ideal scenario: every point matches all 15 neighbors
        current_match_score = match_score  # Current scenario: actual match score

        # loss = (15 * len(mapped_data) - match_score) / len(mapped_data)
        loss = (max_match_score - current_match_score) / max_match_score
    
            # Ensure the loss is between 0 and 1
        loss_tensor = torch.tensor(loss, device=self.device)

        loss = torch.clamp(loss_tensor, min=0.0, max=1.0)
        return torch.tensor(loss, device=self.device)
    
    def transformation_train_advanced(self,lambda_translation=0.5, lambda_similarity=1.0,num_epochs=10,lr=0.001,base_epoch=500):
        original_neighbors = self.compute_neighbors(self.tar_data)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
       
        tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_data, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)

         # Train the autoencoder
        for epoch in range(base_epoch):
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
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Train the adv autoencoder
        for epoch in range(num_epochs):
            self.model.train()

            optimizer.zero_grad()

            # Inside your training loop:
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)
            # neighbor_loss = self.similarity_loss(latent_representation)
            neighbor_loss = self.similarity_loss(latent_representation, original_neighbors)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss + lambda_similarity * neighbor_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # if epoch % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}],reconstruction_loss:{reconstruction_loss.item():.4f},translation_loss:{translation_loss.item():.4f},neighbor_loss:{neighbor_loss.item():.4f}, Loss: {loss.item():.4f}")

        # To get the mapped version of tar
        tar_mapped = self.model.encoder(tar_tensor).cpu().detach().numpy()

        # To get ref back from the mapped tar
        ref_reconstructed = self.model.decoder(torch.tensor(tar_mapped).to(self.device)).cpu().detach().numpy()

        return self.model, tar_mapped, ref_reconstructed
    



    """ 2023.02.22 """
    def _construct_fuzzy_complex(self, train_data, metric="euclidean"):
        # """
        # construct a vietoris-rips complex
        # """
        # number of trees in random projection forest
        n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        # distance metric
        # # get nearest neighbors
        
        nnd = NNDescent(
            train_data,
            n_neighbors=self.n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        knn_indices, knn_dists = nnd.neighbor_graph
        random_state = check_random_state(42)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=self.n_neighbors,
            metric=metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists
        )
        return complex, sigmas, rhos, knn_indices,knn_dists
    
    # def complext_similarity_loss(self, embeddings, knn_indices, knn_dists):
    #     """
    #     Compute the similarity loss that encourages the preservation of the neighborhood structure.

    #     Parameters:
    #     - embeddings: Latent representations of the input data.
    #     - knn_indices: Indices of the K-nearest neighbors for each data point.
    #     - knn_dists: Distances to the K-nearest neighbors for each data point.

    #     Returns:
    #     - A scalar tensor representing the similarity loss.
    #     """
    #     num_points = embeddings.shape[0]
    #     loss = 0.0

    #     for i in range(num_points):
    #         # get the k nearest neibour index and distance
    #         original_neighbors_idx = knn_indices[i]
    #         original_neighbors_dists = knn_dists[i]

    #         # calculate the distance between transformed data and original data
            #   tranxsformed_dists = torch.sqrt(torch.sum((embeddings[i] - embeddings[original_neighbors_idx]) ** 2, dim=1) + 1e-8).to(self.device)

    #         transformed_dists = torch.sqrt(torch.sum((embeddings[i] - embeddings[original_neighbors_idx]) ** 2, dim=1)).to(self.device)

    #         dist_diff = F.relu(transformed_dists - torch.Tensor(original_neighbors_dists).to(self.device))

    #         loss += transformed_dists

    #     # calculate average loss
    #     loss /= num_points
    #     return loss
    def complext_similarity_loss(self, mapped_data, original_neighbors, dists):
        complex, sigmas, rhos, mapped_neighbors ,knn_dists = self._construct_fuzzy_complex(self.tar_data)
    
        mapped_neighbors = self.compute_neighbors(mapped_data.cpu().detach().numpy())

        # calculate the match score
        match_score = 0
        for m_neigh, o_neigh in zip(mapped_neighbors, original_neighbors):
            match_score += len(set(m_neigh) & set(o_neigh))
        
        # most match is 15 
        max_match_score = 15 * len(mapped_data)  # Ideal scenario: every point matches all 15 neighbors
        current_match_score = match_score  # Current scenario: actual match score

        # loss = (15 * len(mapped_data) - match_score) / len(mapped_data)
        loss = (max_match_score - current_match_score) / max_match_score
    
            # Ensure the loss is between 0 and 1
        loss_tensor = torch.tensor(loss, device=self.device)

        loss = torch.clamp(loss_tensor, min=0.0, max=1.0)
        return torch.tensor(loss, device=self.device)
    
    def trans(self,lambda_translation=0.5, lambda_similarity=1.0,num_epochs=100,lr=0.001,base_epoch=10):
        complex, sigmas, rhos, original_neighbors ,knn_dists = self._construct_fuzzy_complex(self.tar_data)

        _, _, _, original_ref_neighbors ,ref_knn_dists = self._construct_fuzzy_complex(self.ref_data)
    
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
       
        tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_data, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)

                 # Train the autoencoder
        for epoch in range(base_epoch):
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
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # if epoch % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Train the adv autoencoder
        for epoch in range(num_epochs):
            self.model.train()

            optimizer.zero_grad()

            # Inside your training loop:
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)
            # neighbor_loss = self.similarity_loss(latent_representation)
            neighbor_loss = self.complext_similarity_loss(latent_representation, original_neighbors,knn_dists)
            neighbor_loss_r = self.complext_similarity_loss(outputs, ref_tensor,ref_knn_dists)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss + lambda_similarity * (neighbor_loss + neighbor_loss_r)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # if epoch % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}],reconstruction_loss:{reconstruction_loss.item():.4f},translation_loss:{translation_loss.item():.4f},neighbor_loss:{neighbor_loss.item():.4f}, Loss: {loss.item():.4f}")

        # To get the mapped version of tar
        tar_mapped = self.model.encoder(tar_tensor).cpu().detach().numpy()

        # To get ref back from the mapped tar
        ref_reconstructed = self.model.decoder(torch.tensor(tar_mapped).to(self.device)).cpu().detach().numpy()

        return self.model, tar_mapped, ref_reconstructed
    


    

