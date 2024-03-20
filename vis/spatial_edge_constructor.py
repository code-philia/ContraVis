from abc import ABC, abstractmethod
import numpy as np
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from vis.backend import get_graph_elements, get_attention

class SpatialEdgeConstructorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider) -> None:
        pass

    @abstractmethod
    def construct(self, *args, **kwargs):
        # return head, tail, weight, feature_vectors
        pass
    
class SpatialEdgeConstructor(SpatialEdgeConstructorAbstractClass):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        """Init parameters for spatial edge constructor

        Parameters
        ----------
        data_provider : data.DataProvider
             data provider
        init_num : int
            init number to calculate c
        s_n_epochs : int
            the number of epochs to fit for one iteration(epoch)
            e.g. n_epochs=5 means each edge will be sampled 5*prob times in one training epoch
        b_n_epochs : int
            the number of epochs to fit boundary samples for one iteration (epoch)
        n_neighbors: int
            local connectivity
        """
        self.data_provider = data_provider
        self.init_num = init_num
        self.s_n_epochs = s_n_epochs
        self.b_n_epochs = b_n_epochs
        self.n_neighbors = n_neighbors
    
    def _construct_fuzzy_complex(self, train_data):
        """
        construct a vietoris-rips complex
        """
        # number of trees in random projection forest
        n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        # distance metric
        metric = "euclidean"
        # get nearest neighbors
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
        random_state = check_random_state(0)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=self.n_neighbors,
            metric=metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
           
        return complex, sigmas, rhos, knn_indices
    
    def _construct_boundary_wise_complex(self, train_data, border_centers):
        """compute the boundary wise complex
            for each border point, we calculate its k nearest train points
            for each train data, we calculate its k nearest border points
        """
        high_neigh = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        high_neigh.fit(border_centers)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        knn_dists, knn_indices = high_neigh.kneighbors(fitting_data, n_neighbors=self.n_neighbors, return_distance=True)
        knn_indices = knn_indices + len(train_data)

        random_state = check_random_state(None)
        bw_complex, sigmas, rhos = fuzzy_simplicial_set(
            X=fitting_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return bw_complex, sigmas, rhos, knn_indices
    
    def _construct_target_wise_complex(self, ref_train_data, tar_train_data):
        """Compute the target-wise complex.
        For each point in ref_train_data, we calculate its k nearest neighbors in tar_train_data 
        and vice versa.
        """

        # Fit the NearestNeighbors on tar_train_data to find neighbors for ref_train_data
        high_neigh_tar = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        high_neigh_tar.fit(tar_train_data)
        knn_dists_ref, knn_indices_ref = high_neigh_tar.kneighbors(ref_train_data, n_neighbors=self.n_neighbors, return_distance=True)

        # Fit the NearestNeighbors on ref_train_data to find neighbors for tar_train_data
        high_neigh_ref = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        high_neigh_ref.fit(ref_train_data)
        knn_dists_tar, knn_indices_tar = high_neigh_ref.kneighbors(tar_train_data, n_neighbors=self.n_neighbors, return_distance=True)

        # Combine the data for fuzzy_simplicial_set (assuming it requires this combination)
        combined_knn_dists = np.vstack((knn_dists_ref, knn_dists_tar))
        combined_knn_indices = np.vstack((knn_indices_ref, knn_indices_tar + len(ref_train_data)))

        fitting_data = np.concatenate((ref_train_data, tar_train_data), axis=0)

        random_state = check_random_state(None)
        tw_complex, sigmas, rhos = fuzzy_simplicial_set(
            X=fitting_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=random_state,
            knn_indices=combined_knn_indices,
            knn_dists=combined_knn_dists,
        )
        return tw_complex, sigmas, rhos, combined_knn_indices
    
    
    def _construct_step_edge_dataset(self, vr_complex, bw_complex):
        """
        construct the mixed edge dataset for one time step
            connect border points and train data(both direction)
        :param vr_complex: Vietoris-Rips complex
        :param bw_complex: boundary-augmented complex
        :param n_epochs: the number of epoch that we iterate each round
        :return: edge dataset
        """
        # get data from graph
        _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        
        # get data from graph
        if self.b_n_epochs == 0:
            return vr_head, vr_tail, vr_weight
        else:
            _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex, self.b_n_epochs)
            head = np.concatenate((vr_head, bw_head), axis=0)
            tail = np.concatenate((vr_tail, bw_tail), axis=0)
            weight = np.concatenate((vr_weight, bw_weight), axis=0)
        return head, tail, weight
    
    def _construct_single_complex_dataset(self, complex, offset):
        """
        Construct the edge dataset for one complex
        """

        def extract_from_complex(complex_data, n_epochs,offset=0):
            # This internal function extracts needed information from the provided complex data
            _, head, tail, weight, _ = get_graph_elements(complex_data, n_epochs,offset)
            return head, tail, weight
        
        head, tail, weight = extract_from_complex(complex, self.s_n_epochs,offset)
       

        return head, tail, weight



    def _construct_edge_dataset(self, complex_ref, complex_tar, tw_complex,offset=0):
        """
        Construct the mixed edge dataset combining three complexes.
        :param complex_ref: Complex for the reference data
        :param complex_tar: Complex for the target data
        :param tw_complex: Target-wise complex
        :return: edge_to, edge_from, weight
        """

        def extract_from_complex(complex_data, n_epochs,offset=0):
            # This internal function extracts needed information from the provided complex data
            _, head, tail, weight, _ = get_graph_elements(complex_data, n_epochs,offset)
            return head, tail, weight

        # Extract data from each complex
        ref_head, ref_tail, ref_weight = extract_from_complex(complex_ref, self.s_n_epochs)
        tar_head, tar_tail, tar_weight = extract_from_complex(complex_tar, self.s_n_epochs,offset)
        
         
        if tw_complex != None:

            tw_head, tw_tail, tw_weight = extract_from_complex(tw_complex, self.b_n_epochs)

            # Concatenate extracted data from all complexes
            edge_from = np.concatenate((ref_head, tar_head, tw_head), axis=0)
            edge_to = np.concatenate((ref_tail, tar_tail, tw_tail), axis=0)
            weight = np.concatenate((ref_weight, tar_weight, tw_weight), axis=0)
        else:
            print('tw complex none')
            edge_from = np.concatenate((ref_head, tar_head), axis=0)
            edge_to = np.concatenate((ref_tail, tar_tail), axis=0)
            weight = np.concatenate((ref_weight, tar_weight), axis=0)

        return edge_to, edge_from, weight


class AlignedSpitalEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, epoch, s_n_epochs, b_n_epochs, n_neighbors, transed_tar, tar_provider) -> None:
        super().__init__(data_provider, epoch, s_n_epochs, b_n_epochs, n_neighbors)
        self.epoch = epoch
        self.trans_tar = transed_tar
        self.tar_provider = tar_provider
        self.iteration = epoch

    def construct(self):
        # load reference and targte train data
        train_data = self.data_provider.train_representation(self.epoch)
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])
        # train_data = np.concatenate((train_data,  self.trans_tar),axis=0)

        # if self.b_n_epochs == 0:
        """if we do not consider the boundary sample"""
        complex,_,_,_ = self._construct_fuzzy_complex(train_data)
        complex_tar,_,_,_ = self._construct_fuzzy_complex(self.trans_tar)
        # tw_complex,_,_,_ =self._construct_target_wise_complex(train_data, self.trans_tar)
        edge_to, edge_from, weight = self._construct_edge_dataset(complex,complex_tar,None,offset=len(train_data))

        feature_vectors = np.concatenate((train_data, self.trans_tar), axis=0)
        
        pred_model = self.data_provider.prediction_function(self.iteration)
        attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)  
        return edge_to, edge_from, weight, feature_vectors, attention
