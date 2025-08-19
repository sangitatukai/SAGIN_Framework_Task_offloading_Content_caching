# rl_formulation_sagin.py - Complete RL Agents for SAGIN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRUTemporalEncoder(nn.Module):
    """GRU-based temporal encoder for capturing IoT activation patterns"""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU for temporal pattern capture
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Feature projectors
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        self.hidden_state = None

    def create_observation_vector(self, zipf_param: float, active_devices: List[int],
                                  content_generation: Dict, cache_hit_rate: float,
                                  energy_ratio: float, queue_length_ratio: float) -> torch.Tensor:
        """Create observation vector"""
        obs_components = []

        # Zipf parameter
        obs_components.append(zipf_param)

        # Device activation features
        if active_devices:
            obs_components.extend([
                len(active_devices) / 20.0,
                np.mean(active_devices) / 20.0,
                np.std(active_devices) / 20.0 if len(active_devices) > 1 else 0.0
            ])
        else:
            obs_components.extend([0.0, 0.0, 0.0])

        # Content generation features
        content_sizes = list(content_generation.values()) if content_generation else [0.0]
        obs_components.extend([
            len(content_generation) / 10.0,
            np.mean(content_sizes) / 20.0,
            np.std(content_sizes) / 20.0 if len(content_sizes) > 1 else 0.0
        ])

        # System state
        obs_components.extend([cache_hit_rate, energy_ratio, queue_length_ratio])

        # Pad to fixed size
        while len(obs_components) < self.input_dim:
            obs_components.append(0.0)
        obs_components = obs_components[:self.input_dim]

        return torch.FloatTensor(obs_components)

    def forward(self, observations: torch.Tensor, reset_hidden: bool = False) -> torch.Tensor:
        """Forward pass through GRU"""
        if reset_hidden:
            self.hidden_state = None

        if observations.dim() == 2:
            observations = observations.unsqueeze(1)

        batch_size = observations.size(0)

        if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
            self.hidden_state = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=observations.device, dtype=observations.dtype
            )

        projected_obs = self.feature_projector(observations)
        gru_output, self.hidden_state = self.gru(projected_obs, self.hidden_state)
        temporal_embedding = gru_output[:, -1, :]

        return temporal_embedding

    def reset_hidden_state(self):
        """Reset hidden state for new episode"""
        self.hidden_state = None


class GraphNeuralNetworkEncoder(nn.Module):
    """Simple GNN encoder for UAV neighbor coordination"""

    def __init__(self, node_feature_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.output_dim = output_dim

        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(node_feature_dim * 2, node_feature_dim),
            nn.ReLU(),
            nn.Linear(node_feature_dim, node_feature_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_feature_dim * 2, node_feature_dim),
            nn.ReLU(),
            nn.Linear(node_feature_dim, output_dim)
        )

    def build_uav_graph(self, uav_positions: Dict[Tuple[int, int], Tuple[float, float, float]]) -> Tuple[
        torch.Tensor, Dict]:
        """Build adjacency matrix for UAV graph"""
        uav_coords = list(uav_positions.keys())
        node_mapping = {coord: i for i, coord in enumerate(uav_coords)}

        edges = []
        communication_range = 200.0

        for i, coord1 in enumerate(uav_coords):
            pos1 = np.array(uav_positions[coord1])
            for j, coord2 in enumerate(uav_coords):
                if i != j:
                    pos2 = np.array(uav_positions[coord2])
                    distance = np.linalg.norm(pos1 - pos2)

                    if distance <= communication_range:
                        edges.append([i, j])

        if edges:
            edge_index = torch.tensor(edges).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return edge_index, node_mapping

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN"""
        if edge_index.size(1) == 0:
            return self.update_mlp(torch.cat([node_features, torch.zeros_like(node_features)], dim=1))

        # Message passing
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        edge_features = torch.cat([source_features, target_features], dim=1)
        messages = self.message_mlp(edge_features)

        # Aggregate messages
        num_nodes = node_features.size(0)
        aggregated = torch.zeros((num_nodes, self.node_feature_dim), device=node_features.device)

        for i in range(edge_index.size(1)):
            target_idx = target_nodes[i]
            aggregated[target_idx] += messages[i]

        # Update node features
        updated_input = torch.cat([node_features, aggregated], dim=1)
        updated_features = self.update_mlp(updated_input)

        return updated_features


class IoTAggregationAgent(nn.Module):
    """Single-agent PPO for IoT device selection"""

    def __init__(self, temporal_dim: int = 64, max_devices: int = 20):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.max_devices = max_devices

        # Device feature processor
        self.device_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(temporal_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(temporal_dim + max_devices * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def create_device_features(self, device_id: int, size: float, ttl: float, transmission_time: float) -> torch.Tensor:
        """Create feature vector for a device"""
        features = torch.FloatTensor([
            size / 20.0,
            ttl / 1800.0,
            transmission_time / 300.0,
            device_id / 20.0
        ])
        return features

    def forward(self, temporal_embedding: torch.Tensor, device_features: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Forward pass for IoT aggregation decisions"""
        batch_size = temporal_embedding.size(0)

        if not device_features:
            return torch.zeros(batch_size, 0), torch.zeros(batch_size, 1)

        device_embeddings = []
        action_probs = []

        for device_feat in device_features:
            if device_feat.dim() == 1:
                device_feat = device_feat.unsqueeze(0).expand(batch_size, -1)

            device_emb = self.device_encoder(device_feat)
            device_embeddings.append(device_emb)

            combined = torch.cat([temporal_embedding, device_emb], dim=1)
            prob = torch.sigmoid(self.actor(combined))
            action_probs.append(prob)

        action_probs = torch.cat(action_probs, dim=1)

        # State value estimation
        all_device_embs = torch.zeros(batch_size, self.max_devices, 16)
        for i, emb in enumerate(device_embeddings[:self.max_devices]):
            all_device_embs[:, i] = emb

        critic_input = torch.cat([temporal_embedding.unsqueeze(1).expand(-1, self.max_devices, -1), all_device_embs],
                                 dim=2)
        critic_input = critic_input.view(batch_size, -1)
        state_value = self.critic(critic_input)

        return action_probs, state_value


class MAPPOCachingOffloadingAgent(nn.Module):
    """Multi-Agent PPO for cooperative caching and task offloading"""

    def __init__(self, temporal_dim: int = 64, spatial_dim: int = 64, max_neighbors: int = 4):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.max_neighbors = max_neighbors

        # Task feature encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Content feature encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Offloading actor
        self.offload_actor = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # [local, neighbor1-4, satellite, drop]
        )

        # Caching actor
        self.cache_actor = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def create_task_burst_features(self, task_burst: List[Dict], energy_ratio: float,
                                   queue_ratio: float) -> torch.Tensor:
        """Create features for task burst"""
        if not task_burst:
            return torch.zeros(1, 8)

        total_cpu = sum(task.get('required_cpu', 1) for task in task_burst)
        min_delay_bound = min(task.get('delay_bound', 10.0) for task in task_burst)
        avg_content_size = np.mean([task.get('size', 1.0) for task in task_burst])

        features = torch.FloatTensor([
            len(task_burst) / 10.0,
            total_cpu / 100.0,
            min_delay_bound / 20.0,
            avg_content_size / 20.0,
            energy_ratio,
            queue_ratio,
            0.0,
            0.0
        ]).unsqueeze(0)

        return features

    def create_content_features(self, content_item: Dict) -> torch.Tensor:
        """Create features for content item"""
        features = torch.FloatTensor([
            content_item.get('size', 1.0) / 20.0,
            content_item.get('ttl', 1200.0) / 1800.0,
            content_item.get('usefulness', 0.5),
            content_item.get('origin', 0)
        ])
        return features

    def forward_offloading(self, temporal_embedding: torch.Tensor, spatial_embedding: torch.Tensor,
                           task_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for offloading decisions"""
        combined = torch.cat([temporal_embedding, spatial_embedding, task_features], dim=-1)
        logits = self.offload_actor(combined)
        return F.softmax(logits, dim=-1)

    def forward_caching(self, temporal_embedding: torch.Tensor, spatial_embedding: torch.Tensor,
                        content_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for caching decisions"""
        combined = torch.cat([temporal_embedding, spatial_embedding, content_features], dim=-1)
        logits = self.cache_actor(combined)
        return torch.sigmoid(logits)

    def forward_critic(self, temporal_embedding: torch.Tensor, spatial_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass for value estimation"""
        combined = torch.cat([temporal_embedding, spatial_embedding], dim=-1)
        return self.critic(combined)


class CentralizedOFDMAgent(nn.Module):
    """Centralized PPO agent for OFDM slot allocation"""

    def __init__(self, max_uavs: int = 9, max_ofdm_slots: int = 6):
        super().__init__()
        self.max_uavs = max_uavs
        self.max_ofdm_slots = max_ofdm_slots

        # Global state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(max_uavs * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, max_uavs)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for OFDM allocation"""
        encoded_state = self.state_encoder(global_state)
        logits = self.actor(encoded_state)
        state_value = self.critic(encoded_state)

        return logits, state_value

    def sample_actions(self, logits: torch.Tensor, num_slots: int) -> torch.Tensor:
        """Sample actions using top-k sampling"""
        batch_size = logits.size(0)
        actions = torch.zeros_like(logits)

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = logits + gumbel_noise

        _, top_k_indices = torch.topk(noisy_logits, num_slots, dim=1)

        for i in range(batch_size):
            actions[i, top_k_indices[i]] = 1

        return actions


class HierarchicalSAGINAgent(nn.Module):
    """Complete hierarchical RL agent combining all sub-agents"""

    def __init__(self, grid_size: Tuple[int, int] = (3, 3), max_devices: int = 20,
                 max_content_items: int = 50, max_neighbors: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.max_devices = max_devices
        self.max_content_items = max_content_items
        self.max_neighbors = max_neighbors

        # Shared temporal encoder
        self.temporal_encoder = GRUTemporalEncoder(input_dim=16, hidden_dim=64)

        # Spatial coordination encoder
        self.spatial_encoder = GraphNeuralNetworkEncoder(node_feature_dim=128, output_dim=64)

        # Individual agents for each UAV
        num_uavs = grid_size[0] * grid_size[1]
        self.iot_agents = nn.ModuleDict({
            f"{x}_{y}": IoTAggregationAgent(temporal_dim=64, max_devices=max_devices)
            for x in range(grid_size[0]) for y in range(grid_size[1])
        })

        self.mappo_agents = nn.ModuleDict({
            f"{x}_{y}": MAPPOCachingOffloadingAgent(temporal_dim=64, spatial_dim=64, max_neighbors=max_neighbors)
            for x in range(grid_size[0]) for y in range(grid_size[1])
        })

        # Centralized OFDM agent
        self.ofdm_agent = CentralizedOFDMAgent(max_uavs=num_uavs, max_ofdm_slots=6)

        # Optimizers
        self.iot_optimizer = torch.optim.Adam(
            list(self.temporal_encoder.parameters()) +
            list(self.iot_agents.parameters()), lr=3e-4
        )

        self.mappo_optimizer = torch.optim.Adam(
            list(self.temporal_encoder.parameters()) +
            list(self.spatial_encoder.parameters()) +
            list(self.mappo_agents.parameters()), lr=3e-4
        )

        self.ofdm_optimizer = torch.optim.Adam(self.ofdm_agent.parameters(), lr=2e-4)

    def reset_temporal_states(self):
        """Reset all temporal states for new episode"""
        self.temporal_encoder.reset_hidden_state()

    def step_iot_aggregation(self, uav_states: Dict, active_devices: Dict, device_contents: Dict) -> Dict:
        """Execute IoT aggregation step for all UAVs"""
        selected_devices = {}

        for (x, y), state in uav_states.items():
            agent_key = f"{x}_{y}"

            if agent_key not in self.iot_agents:
                selected_devices[(x, y)] = []
                continue

            # Create temporal observation
            temporal_obs = self.temporal_encoder.create_observation_vector(
                zipf_param=state.get('zipf_param', 1.5),
                active_devices=active_devices.get((x, y), []),
                content_generation=state.get('content_generation', {}),
                cache_hit_rate=state.get('cache_hit_rate', 0.0),
                energy_ratio=state.get('energy_ratio', 1.0),
                queue_length_ratio=state.get('queue_ratio', 0.0)
            )

            # Get temporal embedding
            temporal_embedding = self.temporal_encoder(temporal_obs.unsqueeze(0))

            # Create device features
            region_devices = active_devices.get((x, y), [])
            region_contents = device_contents.get((x, y), {})

            device_features = []
            for device_id in region_devices:
                if device_id in region_contents:
                    content = region_contents[device_id]
                    features = self.iot_agents[agent_key].create_device_features(
                        device_id=device_id,
                        size=content.get('size', 1.0),
                        ttl=content.get('ttl', 1200.0),
                        transmission_time=0.1
                    )
                    device_features.append(features)

            # Get action probabilities
            if device_features:
                action_probs, _ = self.iot_agents[agent_key](temporal_embedding, device_features)
                action_probs = action_probs.squeeze(0)
                actions = torch.bernoulli(action_probs)
                selected = [region_devices[i] for i, action in enumerate(actions) if action.item() == 1]
                selected_devices[(x, y)] = selected
            else:
                selected_devices[(x, y)] = []

        return selected_devices

    def step_caching_offloading(self, uav_states: Dict, task_bursts: Dict,
                                candidate_content: Dict, uav_positions: Dict) -> Tuple[Dict, Dict]:
        """Execute caching and offloading step for all UAVs"""
        # Build spatial graph
        edge_index, node_mapping = self.spatial_encoder.build_uav_graph(uav_positions)

        # Create node features for all UAVs
        node_features = []
        coord_to_node = {}

        for coord, node_idx in node_mapping.items():
            state = uav_states.get(coord, {})
            coord_to_node[coord] = node_idx

            features = torch.FloatTensor([
                state.get('zipf_param', 1.5),
                state.get('energy_ratio', 1.0),
                state.get('queue_ratio', 0.0),
                state.get('cache_hit_rate', 0.0)
            ])

            padded_features = torch.zeros(128)
            padded_features[:4] = features
            node_features.append(padded_features)

        if node_features:
            node_features = torch.stack(node_features)
            spatial_embeddings = self.spatial_encoder(node_features, edge_index)
        else:
            spatial_embeddings = torch.zeros(0, 64)

        offloading_decisions = {}
        caching_decisions = {}

        for (x, y), state in uav_states.items():
            agent_key = f"{x}_{y}"

            if agent_key not in self.mappo_agents:
                offloading_decisions[(x, y)] = torch.zeros(7)
                caching_decisions[(x, y)] = []
                continue

            # Get embeddings
            temporal_obs = self.temporal_encoder.create_observation_vector(
                zipf_param=state.get('zipf_param', 1.5),
                active_devices=[],
                content_generation={},
                cache_hit_rate=state.get('cache_hit_rate', 0.0),
                energy_ratio=state.get('energy_ratio', 1.0),
                queue_length_ratio=state.get('queue_ratio', 0.0)
            )
            temporal_embedding = self.temporal_encoder(temporal_obs.unsqueeze(0))

            # Get spatial embedding
            if (x, y) in coord_to_node:
                node_idx = coord_to_node[(x, y)]
                spatial_embedding = spatial_embeddings[node_idx:node_idx + 1]
            else:
                spatial_embedding = torch.zeros(1, 64)

            # Task offloading
            tasks = task_bursts.get((x, y), [])
            if tasks:
                task_features = self.mappo_agents[agent_key].create_task_burst_features(
                    tasks, state.get('energy_ratio', 1.0), state.get('queue_ratio', 0.0)
                )
                offload_probs = self.mappo_agents[agent_key].forward_offloading(
                    temporal_embedding, spatial_embedding, task_features
                )

                num_tasks = len(tasks)
                allocation_probs = offload_probs.squeeze(0)
                allocation_counts = torch.multinomial(allocation_probs, num_tasks, replacement=True)

                allocation_tensor = torch.zeros(7)
                for idx in allocation_counts:
                    allocation_tensor[idx] += 1

                offloading_decisions[(x, y)] = allocation_tensor
            else:
                offloading_decisions[(x, y)] = torch.zeros(7)

            # Content caching
            candidates = candidate_content.get((x, y), [])
            selected_content = []

            for content_item in candidates:
                content_features = self.mappo_agents[agent_key].create_content_features(content_item)
                cache_prob = self.mappo_agents[agent_key].forward_caching(
                    temporal_embedding, spatial_embedding, content_features.unsqueeze(0)
                )

                if torch.bernoulli(cache_prob).item() == 1:
                    selected_content.append(content_item)

            caching_decisions[(x, y)] = selected_content

        return offloading_decisions, caching_decisions

    def step_ofdm_allocation(self, uav_states: Dict, max_slots: int = 6) -> Dict:
        """Execute OFDM slot allocation step"""
        # Create global state vector
        global_features = []
        uav_coords = []

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                coord = (x, y)
                uav_coords.append(coord)
                state = uav_states.get(coord, {})

                features = [
                    state.get('aggregated_data', 0.0) / 100.0,
                    state.get('queue_ratio', 0.0),
                    state.get('energy_ratio', 1.0),
                    state.get('cache_hit_rate', 0.0)
                ]
                global_features.extend(features)

        # Pad to max_uavs * 4
        max_features = 9 * 4
        while len(global_features) < max_features:
            global_features.append(0.0)
        global_features = global_features[:max_features]

        global_state = torch.FloatTensor(global_features).unsqueeze(0)

        # Get allocation logits
        logits, _ = self.ofdm_agent(global_state)

        # Sample actions with slot constraint
        actions = self.ofdm_agent.sample_actions(logits, max_slots)

        # Convert to dictionary
        slot_allocation = {}
        for i, coord in enumerate(uav_coords):
            if i < actions.size(1):
                slot_allocation[coord] = bool(actions[0, i].item())
            else:
                slot_allocation[coord] = False

        return slot_allocation

    def update_agents(self, step_rewards: Dict):
        """Update all agents using collected rewards"""
        iot_rewards = step_rewards.get('iot_aggregation', {})
        caching_rewards = step_rewards.get('caching_offloading', {})
        ofdm_rewards = step_rewards.get('ofdm_allocation', [])

        if iot_rewards:
            avg_iot_reward = np.mean(list(iot_rewards.values()))
            logger.debug(f"Average IoT aggregation reward: {avg_iot_reward:.3f}")

        if caching_rewards:
            avg_caching_reward = np.mean(list(caching_rewards.values()))
            logger.debug(f"Average caching/offloading reward: {avg_caching_reward:.3f}")

        if ofdm_rewards:
            avg_ofdm_reward = np.mean(ofdm_rewards)
            logger.debug(f"Average OFDM allocation reward: {avg_ofdm_reward:.3f}")

    def state_dict(self):
        """Return state dict for saving"""
        return {
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'spatial_encoder': self.spatial_encoder.state_dict(),
            'iot_agents': self.iot_agents.state_dict(),
            'mappo_agents': self.mappo_agents.state_dict(),
            'ofdm_agent': self.ofdm_agent.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.temporal_encoder.load_state_dict(state_dict['temporal_encoder'])
        self.spatial_encoder.load_state_dict(state_dict['spatial_encoder'])
        self.iot_agents.load_state_dict(state_dict['iot_agents'])
        self.mappo_agents.load_state_dict(state_dict['mappo_agents'])
        self.ofdm_agent.load_state_dict(state_dict['ofdm_agent'])


# Utility functions for creating and managing RL agents

def create_hierarchical_agent(grid_size: Tuple[int, int] = (3, 3), **kwargs):
    """
    Factory function to create hierarchical SAGIN agent

    Args:
        grid_size: Grid dimensions (X, Y)
        **kwargs: Additional arguments for agent configuration

    Returns:
        Configured HierarchicalSAGINAgent
    """
    return HierarchicalSAGINAgent(
        grid_size=grid_size,
        max_devices=kwargs.get('max_devices', 20),
        max_content_items=kwargs.get('max_content_items', 50),
        max_neighbors=kwargs.get('max_neighbors', 4)
    )


def compute_gae_advantages(rewards: List[float], values: List[float],
                           gamma: float = 0.99, gae_lambda: float = 0.95) -> List[float]:
    """
    Compute Generalized Advantage Estimation (GAE) advantages

    Args:
        rewards: List of rewards
        values: List of value function estimates
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        List of computed advantages
    """
    advantages = []
    gae = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]

        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    return advantages


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Normalize advantages to have zero mean and unit variance"""
    if not advantages:
        return advantages

    advantages = np.array(advantages)
    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages)

    if std_adv > 1e-8:
        normalized = (advantages - mean_adv) / (std_adv + 1e-8)
    else:
        normalized = advantages - mean_adv

    return normalized.tolist()


# Testing and validation functions

def test_hierarchical_agent():
    """Test the hierarchical agent with dummy data"""
    print("🧪 Testing Hierarchical SAGIN Agent")

    # Create agent
    agent = create_hierarchical_agent(grid_size=(2, 2))

    # Test data
    uav_states = {
        (0, 0): {'zipf_param': 1.8, 'energy_ratio': 0.9, 'queue_ratio': 0.1, 'cache_hit_rate': 0.6},
        (0, 1): {'zipf_param': 2.1, 'energy_ratio': 0.8, 'queue_ratio': 0.3, 'cache_hit_rate': 0.4},
        (1, 0): {'zipf_param': 1.5, 'energy_ratio': 0.7, 'queue_ratio': 0.2, 'cache_hit_rate': 0.5},
        (1, 1): {'zipf_param': 1.9, 'energy_ratio': 0.6, 'queue_ratio': 0.4, 'cache_hit_rate': 0.3}
    }

    active_devices = {
        (0, 0): [1, 3, 5],
        (0, 1): [2, 4],
        (1, 0): [1, 2, 6],
        (1, 1): [3, 7, 8]
    }

    device_contents = {
        (0, 0): {1: {'size': 5.0, 'ttl': 1200}, 3: {'size': 8.0, 'ttl': 900}, 5: {'size': 3.0, 'ttl': 1500}},
        (0, 1): {2: {'size': 4.0, 'ttl': 1000}, 4: {'size': 6.0, 'ttl': 800}},
        (1, 0): {1: {'size': 7.0, 'ttl': 1100}, 2: {'size': 2.0, 'ttl': 1300}, 6: {'size': 9.0, 'ttl': 700}},
        (1, 1): {3: {'size': 5.5, 'ttl': 950}, 7: {'size': 3.5, 'ttl': 1400}, 8: {'size': 4.5, 'ttl': 1050}}
    }

    task_bursts = {
        (0, 0): [{'required_cpu': 10, 'delay_bound': 15.0, 'size': 2.0}],
        (0, 1): [{'required_cpu': 8, 'delay_bound': 12.0, 'size': 1.5},
                 {'required_cpu': 15, 'delay_bound': 20.0, 'size': 3.0}],
        (1, 0): [{'required_cpu': 12, 'delay_bound': 18.0, 'size': 2.5}],
        (1, 1): []
    }

    candidate_content = {
        (0, 0): [{'size': 5.0, 'ttl': 1200, 'usefulness': 0.8, 'origin': 0}],
        (0, 1): [{'size': 4.0, 'ttl': 1000, 'usefulness': 0.6, 'origin': 1},
                 {'size': 6.0, 'ttl': 800, 'usefulness': 0.7, 'origin': 0}],
        (1, 0): [{'size': 7.0, 'ttl': 1100, 'usefulness': 0.9, 'origin': 0}],
        (1, 1): [{'size': 5.5, 'ttl': 950, 'usefulness': 0.5, 'origin': 2}]
    }

    uav_positions = {
        (0, 0): (50, 50, 100),
        (0, 1): (50, 150, 100),
        (1, 0): (150, 50, 100),
        (1, 1): (150, 150, 100)
    }

    print("✅ Testing IoT Aggregation...")
    selected_devices = agent.step_iot_aggregation(uav_states, active_devices, device_contents)
    print(f"Selected devices: {selected_devices}")

    print("✅ Testing OFDM Allocation...")
    slot_allocation = agent.step_ofdm_allocation(uav_states, max_slots=3)
    print(f"OFDM allocation: {slot_allocation}")

    print("✅ Testing Caching & Offloading...")
    offloading_decisions, caching_decisions = agent.step_caching_offloading(
        uav_states, task_bursts, candidate_content, uav_positions
    )
    print(f"Offloading decisions: {dict(list(offloading_decisions.items())[:2])}")
    print(f"Caching decisions: {dict(list(caching_decisions.items())[:2])}")

    print("✅ Testing Agent Updates...")
    dummy_rewards = {
        'iot_aggregation': {'0_0': 1.5, '0_1': 2.1, '1_0': 1.8, '1_1': 0.9},
        'caching_offloading': {'0_0': 2.3, '0_1': 1.7, '1_0': 2.8, '1_1': 1.2},
        'ofdm_allocation': [1.9, 2.2, 1.6]
    }
    agent.update_agents(dummy_rewards)

    print("🎉 All tests passed! Hierarchical agent is working correctly.")

    return agent


if __name__ == "__main__":
    # Run tests
    test_agent = test_hierarchical_agent()

    print(f"\n📊 Agent Statistics:")
    print(f"   Total parameters: {sum(p.numel() for p in test_agent.parameters()):,}")
    print(f"   Grid size: {test_agent.grid_size}")
    print(f"   IoT agents: {len(test_agent.iot_agents)}")
    print(f"   MAPPO agents: {len(test_agent.mappo_agents)}")

    # Test state dict functionality
    state_dict = test_agent.state_dict()
    print(f"   State dict keys: {list(state_dict.keys())}")

    print("\n🚀 Hierarchical RL agent ready for integration!")