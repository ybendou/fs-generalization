import random
import torch

class EpisodicGenerator():
    def __init__(self, datasetName=None, dataset_path=None, max_classes=50, num_elements_per_class=None):
        self.dataset = None
        self.num_elements_per_class = num_elements_per_class
        self.max_classes = min(len(self.num_elements_per_class), 50)
                
    def select_classes(self, ways):
        # number of ways for this episode
        n_ways = ways if ways!=0 else random.randint(5, self.max_classes)

        # get n_ways classes randomly
        choices = torch.randperm(len(self.num_elements_per_class))[:n_ways]
        return choices 
    
    def get_query_size(self, choice_classes, n_queries):
        return n_queries

    def get_support_size(self, choice_classes, query_size, n_shots):
        support_size = len(choice_classes)*n_shots
        return support_size
    
    def get_number_of_shots(self, choice_classes, support_size, query_size, n_shots):
        n_shots_per_class = [n_shots]*len(choice_classes)
        return n_shots_per_class

    def get_number_of_queries(self, choice_classes, query_size, unbalanced_queries):
        n_queries_per_class = [query_size]*len(choice_classes)
        return n_queries_per_class

    def sample_indices(self, num_elements_per_chosen_classes, n_shots_per_class, n_queries_per_class):
        shots_idx = []
        queries_idx = []
        for k, q, elements_per_class in zip(n_shots_per_class, n_queries_per_class, num_elements_per_chosen_classes):
            choices = torch.randperm(elements_per_class)
            shots_idx.append(choices[:k].tolist())
            queries_idx.append(choices[k:k+q].tolist())
        return shots_idx, queries_idx

    def sample_episode(self, ways=0, n_shots=0, n_queries=0, unbalanced_queries=False, verbose=False):
        """
        Sample an episode
        """
        # get n_ways classes randomly
        choice_classes = self.select_classes(ways=ways)
        
        query_size = self.get_query_size(choice_classes, n_queries)
        support_size = self.get_support_size(choice_classes, query_size, n_shots)

        n_shots_per_class = self.get_number_of_shots(choice_classes, support_size, query_size, n_shots)
        n_queries_per_class = self.get_number_of_queries(choice_classes, query_size, unbalanced_queries)
        shots_idx, queries_idx = self.sample_indices([self.num_elements_per_class[c] for c in choice_classes], n_shots_per_class, n_queries_per_class)

        if verbose:
            print(f'chosen class: {choice_classes}')
            print(f'n_ways={len(choice_classes)}, q={query_size}, S={support_size}, n_shots_per_class={n_shots_per_class}')
            print(f'queries per class:{n_queries_per_class}')
            print(f'shots_idx: {shots_idx}')
            print(f'queries_idx: {queries_idx}')

        return {'choice_classes':choice_classes, 'shots_idx':shots_idx, 'queries_idx':queries_idx}

    def get_features_from_indices(self, features, episode, validation=False):
        """
        Get features from a list of all features and from a dictonnary describing an episode
        """
        choice_classes, shots_idx, queries_idx = episode['choice_classes'], episode['shots_idx'], episode['queries_idx']
        if validation : 
            validation_idx = episode['validations_idx']
            val = []
        shots, queries = [], []
        for i, c in enumerate(choice_classes):
            shots.append(features[c]['features'][shots_idx[i]])
            queries.append(features[c]['features'][queries_idx[i]])
            if validation : 
                val.append(features[c]['features'][validation_idx[i]])
        if validation:
            return shots, queries, val
        else:
            return shots, queries
