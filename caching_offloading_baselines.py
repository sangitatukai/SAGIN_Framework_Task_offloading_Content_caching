# caching_offloading_baselines.py - Caching & Task Offloading Baseline Approaches
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


class PopularityToSizeGreedyCaching:
    """
    Baseline 1: Popularity-to-size greedy caching
    Ranks candidates by usefulness/size ratio and fills cache until full
    """

    def __init__(self, cache_capacity_mb):
        self.cache_capacity_mb = cache_capacity_mb
        self.content_popularity = defaultdict(float)

    def select_cache_content(self, candidate_pool):
        """
        Select content for caching using popularity-to-size greedy approach

        Args:
            candidate_pool: {cid: metadata} dictionary of candidate content

        Returns:
            selected_content: {cid: metadata} dictionary of selected content
        """
        if not candidate_pool:
            return {}

        # Calculate priority scores: usefulness/size
        content_priorities = []

        for cid, metadata in candidate_pool.items():
            usefulness = self.content_popularity.get(cid, 1.0)  # Default usefulness
            size = metadata.get('size', 1.0)
            priority = usefulness / max(size, 0.1)  # Avoid division by zero
            content_priorities.append((priority, cid, metadata))

        # Sort by priority (descending)
        content_priorities.sort(key=lambda x: x[0], reverse=True)

        # Greedily fill cache until capacity is reached
        selected_content = {}
        used_capacity = 0.0

        for priority, cid, metadata in content_priorities:
            content_size = metadata.get('size', 0)

            if used_capacity + content_size <= self.cache_capacity_mb:
                selected_content[cid] = metadata
                used_capacity += content_size
            else:
                break  # Cache full

        return selected_content

    def update_popularity(self, cid, was_used):
        """Update content popularity based on usage"""
        # Exponential moving average
        self.content_popularity[cid] = 0.8 * int(was_used) + 0.2 * self.content_popularity[cid]


class StatelessPPOCaching(nn.Module):
    """
    Baseline 2: Stateless PPO caching
    Logits depend only on item metadata, ignoring GRU and GNN context
    """

    def __init__(self, item_feature_dim=4, hidden_dim=32):
        super().__init__()
        self.item_encoder = nn.Sequential(
            nn.Linear(item_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary decision per item
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def select_cache_content(self, candidate_pool, cache_capacity_mb):
        """
        Select content using stateless neural network
        """
        if not candidate_pool:
            return {}

        # Convert content to feature vectors
        content_items = list(candidate_pool.items())
        features = []

        for cid, metadata in content_items:
            # Create feature vector: [size, ttl_normalized, popularity, age]
            feature = [
                metadata.get('size', 1.0),
                metadata.get('ttl', 1800) / 1800,  # Normalized TTL
                metadata.get('popularity', 1.0),
                metadata.get('generation_time', 0) / 1000  # Normalized age
            ]
            features.append(feature)

        features_tensor = torch.FloatTensor(features)

        # Get selection probabilities
        with torch.no_grad():
            logits = self.item_encoder(features_tensor).squeeze()
            probs = torch.sigmoid(logits)

        # Greedy selection based on probabilities and capacity constraint
        prob_cid_pairs = [(probs[i].item(), content_items[i]) for i in range(len(content_items))]
        prob_cid_pairs.sort(key=lambda x: x[0], reverse=True)

        selected_content = {}
        used_capacity = 0.0

        for prob, (cid, metadata) in prob_cid_pairs:
            content_size = metadata.get('size', 0)

            if used_capacity + content_size <= cache_capacity_mb:
                selected_content[cid] = metadata
                used_capacity += content_size
            else:
                break

        return selected_content


class RandomSplitTaskOffloading:
    """
    Baseline 1: Random split task offloading
    Uniformly distributes task burst across neighbors and satellite
    """

    def __init__(self):
        pass

    def make_offloading_decisions(self, task_burst, uav_coord, neighbor_coords,
                                  satellite_available=True):
        """
        Randomly distribute tasks across available execution locations

        Args:
            task_burst: List of tasks to offload
            uav_coord: Current UAV coordinates
            neighbor_coords: List of neighbor UAV coordinates
            satellite_available: Whether satellite connection is available

        Returns:
            offloading_decisions: {task_id: target_location}
        """
        if not task_burst:
            return {}

        # Define available execution locations
        execution_options = ['local']
        execution_options.extend([f'neighbor_{coord[0]}_{coord[1]}' for coord in neighbor_coords])

        if satellite_available:
            execution_options.append('satellite')

        # Randomly assign each task
        offloading_decisions = {}

        for task in task_burst:
            target = np.random.choice(execution_options)
            offloading_decisions[task['task_id']] = target

        return offloading_decisions


class GreedyContentAwareOffloading:
    """
    Baseline 2: Greedy content-aware offloading
    Makes decisions based on content availability and queue status
    """

    def __init__(self):
        pass

    def make_offloading_decisions(self, task_burst, uav_coord, uav_state,
                                  neighbor_states, satellite_state):
        """
        Make content-aware offloading decisions

        Args:
            task_burst: List of tasks to offload
            uav_coord: Current UAV coordinates
            uav_state: Current UAV state (cache, queue, energy)
            neighbor_states: {coord: neighbor_state} dictionary
            satellite_state: Satellite state information

        Returns:
            offloading_decisions: {task_id: target_location}
        """
        offloading_decisions = {}

        for task in task_burst:
            cid = tuple(task['content_id'])
            task_id = task['task_id']

            # Option 1: Local execution
            if (cid in uav_state.get('cache', set()) or
                    cid in uav_state.get('aggregated_content', set())):
                if len(uav_state.get('queue', [])) < uav_state.get('max_queue', 10):
                    offloading_decisions[task_id] = 'local'
                    continue

            # Option 2: Best neighbor
            best_neighbor = None
            min_queue_load = float('inf')

            for neighbor_coord, neighbor_state in neighbor_states.items():
                # Check if neighbor has content and capacity
                if (cid in neighbor_state.get('cache', set()) or
                        cid in neighbor_state.get('aggregated_content', set())):

                    queue_load = len(neighbor_state.get('queue', []))
                    energy_ratio = neighbor_state.get('energy_ratio', 0)

                    if (queue_load < neighbor_state.get('max_queue', 10) and
                            energy_ratio > 0.2 and queue_load < min_queue_load):
                        min_queue_load = queue_load
                        best_neighbor = neighbor_coord

            if best_neighbor is not None:
                offloading_decisions[task_id] = f'neighbor_{best_neighbor[0]}_{best_neighbor[1]}'
                continue

            # Option 3: Satellite
            if (satellite_state.get('available', False) and
                    cid in satellite_state.get('global_content_pool', set())):
                offloading_decisions[task_id] = 'satellite'
                continue

            # Option 4: Drop task
            offloading_decisions[task_id] = 'dropped'

        return offloading_decisions


class LoadBalancedOffloading:
    """
    Baseline 3: Load-balanced offloading
    Distributes tasks to minimize queue imbalance
    """

    def __init__(self):
        pass

    def make_offloading_decisions(self, task_burst, uav_coord, uav_state,
                                  neighbor_states, satellite_state):
        """
        Make load-balanced offloading decisions
        """
        if not task_burst:
            return {}

        offloading_decisions = {}

        # Calculate current loads
        execution_loads = {'local': len(uav_state.get('queue', []))}

        for neighbor_coord, neighbor_state in neighbor_states.items():
            neighbor_key = f'neighbor_{neighbor_coord[0]}_{neighbor_coord[1]}'
            execution_loads[neighbor_key] = len(neighbor_state.get('queue', []))

        if satellite_state.get('available', False):
            execution_loads['satellite'] = satellite_state.get('queue_length', 0)

        # Assign tasks to least loaded execution location
        for task in task_burst:
            cid = tuple(task['content_id'])
            task_id = task['task_id']

            # Find locations that have the required content
            available_locations = []

            # Check local
            if (cid in uav_state.get('cache', set()) or
                    cid in uav_state.get('aggregated_content', set())):
                available_locations.append('local')

            # Check neighbors
            for neighbor_coord, neighbor_state in neighbor_states.items():
                if (cid in neighbor_state.get('cache', set()) or
                        cid in neighbor_state.get('aggregated_content', set())):
                    neighbor_key = f'neighbor_{neighbor_coord[0]}_{neighbor_coord[1]}'
                    available_locations.append(neighbor_key)

            # Check satellite
            if (satellite_state.get('available', False) and
                    cid in satellite_state.get('global_content_pool', set())):
                available_locations.append('satellite')

            if available_locations:
                # Choose least loaded location among available ones
                min_load = float('inf')
                best_location = None

                for location in available_locations:
                    if execution_loads.get(location, float('inf')) < min_load:
                        min_load = execution_loads[location]
                        best_location = location

                offloading_decisions[task_id] = best_location
                execution_loads[best_location] = execution_loads.get(best_location, 0) + 1
            else:
                # No location has content
                offloading_decisions[task_id] = 'dropped'

        return offloading_decisions


# Factory functions
def create_caching_baseline(baseline_type, **kwargs):
    """Factory function for caching baselines"""
    if baseline_type == 'greedy':
        return PopularityToSizeGreedyCaching(**kwargs)
    elif baseline_type == 'stateless_ppo':
        return StatelessPPOCaching(**kwargs)
    else:
        raise ValueError(f"Unknown caching baseline: {baseline_type}")


def create_offloading_baseline(baseline_type, **kwargs):
    """Factory function for offloading baselines"""
    if baseline_type == 'random_split':
        return RandomSplitTaskOffloading(**kwargs)
    elif baseline_type == 'content_aware':
        return GreedyContentAwareOffloading(**kwargs)
    elif baseline_type == 'load_balanced':
        return LoadBalancedOffloading(**kwargs)
    else:
        raise ValueError(f"Unknown offloading baseline: {baseline_type}")


# Testing
if __name__ == "__main__":
    print("=== Testing Caching & Offloading Baselines ===")

    # Test caching baselines
    print("\n--- Testing Caching Baselines ---")

    candidate_pool = {
        (0, 0, 1): {'size': 5.0, 'ttl': 1200, 'popularity': 0.8},
        (0, 0, 2): {'size': 3.0, 'ttl': 900, 'popularity': 0.6},
        (1, 1, 3): {'size': 8.0, 'ttl': 1500, 'popularity': 0.9},
        (2, 2, 4): {'size': 12.0, 'ttl': 600, 'popularity': 0.4}
    }

    cache_capacity = 10.0  # MB

    # Test greedy caching
    greedy_cache = create_caching_baseline('greedy', cache_capacity_mb=cache_capacity)
    selected = greedy_cache.select_cache_content(candidate_pool)
    print(f"Greedy caching selected: {len(selected)} items")
    total_size = sum(item['size'] for item in selected.values())
    print(f"Total cache size used: {total_size:.1f}/{cache_capacity}MB")

    # Test offloading baselines
    print("\n--- Testing Offloading Baselines ---")

    task_burst = [
        {'task_id': 1, 'content_id': (0, 0, 1)},
        {'task_id': 2, 'content_id': (1, 1, 3)},
        {'task_id': 3, 'content_id': (2, 2, 4)}
    ]

    uav_coord = (1, 1)
    neighbor_coords = [(0, 1), (2, 1), (1, 0), (1, 2)]

    # Test random split
    random_offloader = create_offloading_baseline('random_split')
    decisions = random_offloader.make_offloading_decisions(
        task_burst, uav_coord, neighbor_coords, satellite_available=True
    )
    print(f"Random split decisions: {decisions}")

    print("\n=== Caching & Offloading Baselines Test Complete ===")