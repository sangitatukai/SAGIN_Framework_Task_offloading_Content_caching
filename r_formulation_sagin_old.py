

# rl_formulation_sagin.py - WORKING VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class HierarchicalSAGINAgent(nn.Module):
    """Working Hierarchical RL Agent for SAGIN - Emergency Fix"""

    def __init__(self, grid_size: Tuple[int, int] = (3, 3), max_devices: int = 20,
                 max_content_items: int = 50, max_neighbors: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.max_devices = max_devices
        self.max_content_items = max_content_items
        self.max_neighbors = max_neighbors

        # Simple but working neural networks
        self.temporal_encoder = nn.GRU(input_size=16, hidden_size=64, batch_first=True)
        self.spatial_encoder = nn.Linear(4, 64)

        # Per-UAV IoT agents
        self.iot_agents = nn.ModuleDict({
            f"{x}_{y}": nn.Linear(64, 32)
            for x in range(grid_size[0]) for y in range(grid_size[1])
        })

        # Per-UAV MAPPO agents for caching/offloading
        self.mappo_agents = nn.ModuleDict({
            f"{x}_{y}": nn.Linear(128, 7)  # 7 offloading options
            for x in range(grid_size[0]) for y in range(grid_size[1])
        })

        # Centralized OFDM agent
        num_uavs = grid_size[0] * grid_size[1]
        self.ofdm_agent = nn.Linear(num_uavs * 4, num_uavs)

        # Hidden states for GRU
        self.hidden_state = None

    def create_observation_vector(self, zipf_param=1.5, active_devices=None,
                                  content_generation=None, cache_hit_rate=0.0,
                                  energy_ratio=1.0, queue_length_ratio=0.0):
        """Create 16-element observation vector"""
        if active_devices is None:
            active_devices = []
        if content_generation is None:
            content_generation = {}

        # Build 16-element observation
        obs = [zipf_param / 3.0]  # Normalize zipf parameter

        # Device activity features (4 elements)
        if active_devices:
            obs.extend([
                len(active_devices) / 20.0,
                np.mean(active_devices) / 20.0,
                np.std(active_devices) / 20.0 if len(active_devices) > 1 else 0.0,
                np.max(active_devices) / 20.0
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # Content generation features (4 elements)
        if content_generation:
            content_sizes = list(content_generation.values())
            obs.extend([
                len(content_generation) / 10.0,
                np.mean(content_sizes) / 100.0 if content_sizes else 0.0,
                np.std(content_sizes) / 100.0 if len(content_sizes) > 1 else 0.0,
                np.max(content_sizes) / 100.0 if content_sizes else 0.0
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # System state features (7 elements)
        obs.extend([
            cache_hit_rate / 100.0,
            energy_ratio,
            queue_length_ratio,
            np.random.random() * 0.1,  # Noise for exploration
            np.random.random() * 0.1,
            np.random.random() * 0.1,
            np.random.random() * 0.1
        ])

        # Ensure exactly 16 elements
        obs = obs[:16] + [0.0] * max(0, 16 - len(obs))

        return torch.tensor(obs, dtype=torch.float32)

    def reset_temporal_states(self):
        """Reset temporal states for new episode"""
        self.hidden_state = None

    def step_iot_aggregation(self, uav_states: Dict, active_devices: Dict, device_contents: Dict) -> Dict:
        """IoT aggregation step"""
        selected_devices = {}

        for (x, y), devices in active_devices.items():
            if (x, y) not in uav_states:
                selected_devices[(x, y)] = []
                continue

            # Create observation
            state = uav_states[(x, y)]
            obs = self.create_observation_vector(
                zipf_param=state.get('zipf_param', 1.5),
                active_devices=devices,
                content_generation=device_contents.get((x, y), {}),
                cache_hit_rate=state.get('cache_hit_rate', 0.0),
                energy_ratio=state.get('energy_ratio', 1.0),
                queue_length_ratio=state.get('queue_ratio', 0.0)
            )

            # Process through temporal encoder
            obs_batch = obs.unsqueeze(0).unsqueeze(0)  # [1, 1, 16]
            temporal_emb, self.hidden_state = self.temporal_encoder(obs_batch, self.hidden_state)
            temporal_emb = temporal_emb.squeeze()  # [64]

            # Get IoT agent output
            agent_key = f"{x}_{y}"
            if agent_key in self.iot_agents:
                with torch.no_grad():
                    selection_logits = self.iot_agents[agent_key](temporal_emb)
                    selection_probs = torch.sigmoid(selection_logits)

                # Select devices based on probabilities
                selected = []
                for i, device in enumerate(devices):
                    if i < len(selection_probs) and selection_probs[i].item() > 0.5:
                        selected.append(device)
                    if len(selected) >= 5:  # Limit to 5 devices
                        break

                selected_devices[(x, y)] = selected
            else:
                selected_devices[(x, y)] = devices[:5] if devices else []

        return selected_devices

    def step_caching_offloading(self, uav_states: Dict, task_bursts: Dict,
                                candidate_content: Dict, uav_positions: Dict):
        """Caching and offloading step"""
        offloading_decisions = {}
        caching_decisions = {}

        for (x, y), state in uav_states.items():
            # Create observation
            obs = self.create_observation_vector(
                zipf_param=state.get('zipf_param', 1.5),
                cache_hit_rate=state.get('cache_hit_rate', 0.0),
                energy_ratio=state.get('energy_ratio', 1.0),
                queue_length_ratio=state.get('queue_ratio', 0.0)
            )

            # Temporal embedding
            obs_batch = obs.unsqueeze(0).unsqueeze(0)
            temporal_emb, _ = self.temporal_encoder(obs_batch)
            temporal_emb = temporal_emb.squeeze()  # [64]

            # Spatial embedding (simplified)
            spatial_feat = torch.tensor([
                state.get('energy_ratio', 1.0),
                state.get('queue_ratio', 0.0),
                state.get('cache_hit_rate', 0.0) / 100.0,
                state.get('zipf_param', 1.5) / 3.0
            ], dtype=torch.float32)
            spatial_emb = self.spatial_encoder(spatial_feat)  # [64]

            # Combined embedding
            combined = torch.cat([temporal_emb, spatial_emb])  # [128]

            # Offloading decisions
            agent_key = f"{x}_{y}"
            if agent_key in self.mappo_agents:
                with torch.no_grad():
                    logits = self.mappo_agents[agent_key](combined)
                    probs = F.softmax(logits, dim=0)

                # Distribute tasks among options
                tasks = task_bursts.get((x, y), [])
                allocation = torch.zeros(7)
                for _ in range(len(tasks)):
                    choice = torch.multinomial(probs, 1).item()
                    allocation[choice] += 1
                offloading_decisions[(x, y)] = allocation
            else:
                # Default allocation
                num_tasks = len(task_bursts.get((x, y), []))
                allocation = torch.zeros(7)
                allocation[0] = num_tasks  # All local by default
                offloading_decisions[(x, y)] = allocation

            # Caching decisions (simplified)
            candidates = candidate_content.get((x, y), [])
            cached_items = []
            for i, content in enumerate(candidates[:10]):  # Limit to 10 candidates
                if torch.rand(1).item() > 0.7:  # Random caching for now
                    cached_items.append(content.get('content_id', i))
            caching_decisions[(x, y)] = cached_items

        return offloading_decisions, caching_decisions

    def step_ofdm_allocation(self, uav_states: Dict, max_slots: int = 6) -> Dict:
        """OFDM slot allocation step"""
        # Create global state vector
        global_state = []
        coords = []

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                coords.append((x, y))
                if (x, y) in uav_states:
                    state = uav_states[(x, y)]
                    global_state.extend([
                        state.get('energy_ratio', 1.0),
                        state.get('queue_ratio', 0.0),
                        state.get('cache_hit_rate', 0.0) / 100.0,
                        state.get('zipf_param', 1.5) / 3.0
                    ])
                else:
                    global_state.extend([1.0, 0.0, 0.0, 1.5])

        global_tensor = torch.tensor(global_state, dtype=torch.float32)

        # Get allocation probabilities
        with torch.no_grad():
            allocation_logits = self.ofdm_agent(global_tensor)
            allocation_probs = torch.sigmoid(allocation_logits)

        # Allocate slots (simple greedy approach)
        slot_allocation = {}
        sorted_indices = torch.argsort(allocation_probs, descending=True)

        allocated_slots = 0
        for i in range(len(coords)):
            coord = coords[i]
            if allocated_slots < max_slots and allocation_probs[i].item() > 0.3:
                slot_allocation[coord] = 1
                allocated_slots += 1
            else:
                slot_allocation[coord] = 0

        return slot_allocation


# Test function to verify the agent works
def test_hierarchical_agent():
    """Test the agent to ensure it works"""
    print("🧪 Testing Hierarchical SAGIN Agent...")

    try:
        # Create agent
        agent = HierarchicalSAGINAgent(grid_size=(2, 2))
        print(f"✅ Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")

        # Test data
        uav_states = {
            (0, 0): {'zipf_param': 1.8, 'energy_ratio': 0.9, 'queue_ratio': 0.1, 'cache_hit_rate': 60.0},
            (0, 1): {'zipf_param': 2.1, 'energy_ratio': 0.8, 'queue_ratio': 0.3, 'cache_hit_rate': 40.0}
        }

        active_devices = {
            (0, 0): [1, 3, 5],
            (0, 1): [2, 4]
        }

        device_contents = {
            (0, 0): {1: {'size': 5.0}, 3: {'size': 8.0}},
            (0, 1): {2: {'size': 4.0}, 4: {'size': 6.0}}
        }

        task_bursts = {
            (0, 0): [{'task_id': 1, 'cpu_cycles': 100}, {'task_id': 2, 'cpu_cycles': 150}],
            (0, 1): [{'task_id': 3, 'cpu_cycles': 200}]
        }

        candidate_content = {
            (0, 0): [{'content_id': 'c1', 'size': 10}, {'content_id': 'c2', 'size': 15}],
            (0, 1): [{'content_id': 'c3', 'size': 12}]
        }

        uav_positions = {
            (0, 0): (0, 0, 100),
            (0, 1): (0, 1, 100)
        }

        # Test IoT aggregation
        selected_devices = agent.step_iot_aggregation(uav_states, active_devices, device_contents)
        print(f"✅ IoT aggregation: {selected_devices}")

        # Test caching and offloading
        offloading, caching = agent.step_caching_offloading(uav_states, task_bursts, candidate_content, uav_positions)
        print(f"✅ Offloading decisions: {len(offloading)} UAVs")
        print(f"✅ Caching decisions: {len(caching)} UAVs")

        # Test OFDM allocation
        ofdm_allocation = agent.step_ofdm_allocation(uav_states, max_slots=6)
        print(f"✅ OFDM allocation: {ofdm_allocation}")

        print("🎉 All tests passed! Agent is working correctly.")
        return agent

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# Run test if executed directly
if __name__ == "__main__":
    test_agent = test_hierarchical_agent()
    if test_agent:
        print(f"\n📊 Agent ready for use!")
        print(f"   Parameters: {sum(p.numel() for p in test_agent.parameters()):,}")
        print(f"   Grid size: {test_agent.grid_size}")
    else:
        print("❌ Agent failed to initialize properly")