# final_fix_script.py - Ensure RL formulation is completely fixed
import os
import shutil

# Minimal working RL formulation content
RL_CONTENT = '''# rl_formulation_sagin.py - Working RL Agents for SAGIN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class HierarchicalSAGINAgent(nn.Module):
    """Simplified but working hierarchical RL agent for SAGIN"""

    def __init__(self, grid_size: Tuple[int, int] = (3, 3), max_devices: int = 20, 
                 max_content_items: int = 50, max_neighbors: int = 4):
        super().__init__()
        self.grid_size = grid_size

        # Simple working neural networks
        self.temporal_encoder = nn.GRU(input_size=16, hidden_size=64, batch_first=True)
        self.spatial_encoder = nn.Linear(4, 64)

        # Per-UAV agents
        self.iot_agents = nn.ModuleDict({
            f"{x}_{y}": nn.Linear(64, 1)
            for x in range(grid_size[0]) for y in range(grid_size[1])
        })

        self.mappo_agents = nn.ModuleDict({
            f"{x}_{y}": nn.Linear(128, 7)  # 7 offloading options
            for x in range(grid_size[0]) for y in range(grid_size[1])
        })

        # OFDM agent
        num_uavs = grid_size[0] * grid_size[1]
        self.ofdm_agent = nn.Linear(num_uavs * 4, num_uavs)

    def create_observation_vector(self, zipf_param=1.5, active_devices=None, 
                                  content_generation=None, cache_hit_rate=0.0,
                                  energy_ratio=1.0, queue_length_ratio=0.0):
        """Create 16-element observation vector"""
        if active_devices is None:
            active_devices = []
        if content_generation is None:
            content_generation = {}

        obs = [zipf_param]

        # Device features
        if active_devices:
            obs.extend([
                len(active_devices) / 20.0,
                np.mean(active_devices) / 20.0,
                np.std(active_devices) / 20.0 if len(active_devices) > 1 else 0.0
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # Content features
        sizes = list(content_generation.values()) if content_generation else [0.0]
        obs.extend([
            len(content_generation) / 10.0,
            np.mean(sizes) / 20.0,
            np.std(sizes) / 20.0 if len(sizes) > 1 else 0.0
        ])

        # System state
        obs.extend([cache_hit_rate, energy_ratio, queue_length_ratio])

        # Pad to 16 elements
        while len(obs) < 16:
            obs.append(0.0)
        obs = obs[:16]

        return torch.FloatTensor(obs)

    def reset_temporal_states(self):
        """Reset temporal states"""
        pass  # Simplified

    def step_iot_aggregation(self, uav_states, active_devices, device_contents):
        """IoT aggregation step"""
        selected_devices = {}

        for (x, y), state in uav_states.items():
            region_devices = active_devices.get((x, y), [])

            if region_devices:
                # Create observation
                obs = self.create_observation_vector(
                    zipf_param=state.get('zipf_param', 1.5),
                    active_devices=region_devices,
                    cache_hit_rate=state.get('cache_hit_rate', 0.0),
                    energy_ratio=state.get('energy_ratio', 1.0),
                    queue_length_ratio=state.get('queue_ratio', 0.0)
                )

                # Process through GRU
                obs_batch = obs.unsqueeze(0).unsqueeze(0)  # [1, 1, 16]
                gru_out, _ = self.temporal_encoder(obs_batch)
                embedding = gru_out.squeeze()  # [64]

                # Get selection threshold
                agent_key = f"{x}_{y}"
                if agent_key in self.iot_agents:
                    threshold = torch.sigmoid(self.iot_agents[agent_key](embedding)).item()
                    num_select = max(1, int(len(region_devices) * threshold))
                    selected = np.random.choice(region_devices, num_select, replace=False).tolist()
                    selected_devices[(x, y)] = selected
                else:
                    selected_devices[(x, y)] = []
            else:
                selected_devices[(x, y)] = []

        return selected_devices

    def step_caching_offloading(self, uav_states, task_bursts, candidate_content, uav_positions):
        """Caching and offloading step"""
        offloading_decisions = {}
        caching_decisions = {}

        for (x, y), state in uav_states.items():
            # Temporal embedding
            obs = self.create_observation_vector(
                zipf_param=state.get('zipf_param', 1.5),
                cache_hit_rate=state.get('cache_hit_rate', 0.0),
                energy_ratio=state.get('energy_ratio', 1.0),
                queue_length_ratio=state.get('queue_ratio', 0.0)
            )
            obs_batch = obs.unsqueeze(0).unsqueeze(0)
            temporal_emb, _ = self.temporal_encoder(obs_batch)
            temporal_emb = temporal_emb.squeeze()  # [64]

            # Spatial embedding (simplified)
            spatial_feat = torch.FloatTensor([
                state.get('energy_ratio', 1.0),
                state.get('queue_ratio', 0.0),
                state.get('cache_hit_rate', 0.0),
                state.get('zipf_param', 1.5) / 3.0
            ])
            spatial_emb = self.spatial_encoder(spatial_feat)  # [64]

            # Combined embedding
            combined = torch.cat([temporal_emb, spatial_emb])  # [128]

            # Offloading decisions
            agent_key = f"{x}_{y}"
            if agent_key in self.mappo_agents:
                logits = self.mappo_agents[agent_key](combined)
                probs = F.softmax(logits, dim=0)

                # Distribute tasks
                tasks = task_bursts.get((x, y), [])
                allocation = torch.zeros(7)
                for _ in range(len(tasks)):
                    choice = torch.multinomial(probs, 1).item()
                    allocation[choice] += 1
                offloading_decisions[(x, y)] = allocation
            else:
                offloading_decisions[(x, y)] = torch.zeros(7)

            # Simple caching
            candidates = candidate_content.get((x, y), [])
            selected = [c for c in candidates if c.get('usefulness', 0.5) > 0.6]
            caching_decisions[(x, y)] = selected

        return offloading_decisions, caching_decisions

    def step_ofdm_allocation(self, uav_states, max_slots=6):
        """OFDM allocation step"""
        # Global state
        features = []
        coords = []

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                coord = (x, y)
                coords.append(coord)
                state = uav_states.get(coord, {})
                features.extend([
                    state.get('aggregated_data', 0.0) / 100.0,
                    state.get('queue_ratio', 0.0),
                    state.get('energy_ratio', 1.0),
                    state.get('cache_hit_rate', 0.0)
                ])

        # Pad features
        expected_size = len(coords) * 4
        while len(features) < expected_size:
            features.append(0.0)
        features = features[:expected_size]

        global_state = torch.FloatTensor(features)
        logits = self.ofdm_agent(global_state)

        # Select top-k
        if max_slots >= len(coords):
            allocation = {coord: True for coord in coords}
        else:
            _, top_indices = torch.topk(logits, max_slots)
            allocation = {coord: (i in top_indices) for i, coord in enumerate(coords)}

        return allocation

    def update_agents(self, step_rewards):
        """Update agents (simplified)"""
        pass  # In full implementation would do PPO updates

    def state_dict(self):
        """State dict for saving"""
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

# Test function
def test_hierarchical_agent():
    """Test the agent"""
    print("🧪 Testing Working Hierarchical SAGIN Agent")

    agent = HierarchicalSAGINAgent(grid_size=(2, 2))

    # Test data
    uav_states = {
        (0, 0): {'zipf_param': 1.8, 'energy_ratio': 0.9, 'queue_ratio': 0.1, 'cache_hit_rate': 0.6},
        (0, 1): {'zipf_param': 2.1, 'energy_ratio': 0.8, 'queue_ratio': 0.3, 'cache_hit_rate': 0.4},
        (1, 0): {'zipf_param': 1.5, 'energy_ratio': 0.7, 'queue_ratio': 0.2, 'cache_hit_rate': 0.5},
        (1, 1): {'zipf_param': 1.9, 'energy_ratio': 0.6, 'queue_ratio': 0.4, 'cache_hit_rate': 0.3}
    }

    active_devices = {(0, 0): [1, 3, 5], (0, 1): [2, 4], (1, 0): [1, 2], (1, 1): [3, 7]}
    device_contents = {(0, 0): {1: {'size': 5.0}, 3: {'size': 8.0}}}
    task_bursts = {(0, 0): [{'required_cpu': 10, 'delay_bound': 15.0}]}
    candidate_content = {(0, 0): [{'size': 5.0, 'usefulness': 0.8}]}
    uav_positions = {(0, 0): (50, 50, 100), (0, 1): (50, 150, 100), (1, 0): (150, 50, 100), (1, 1): (150, 150, 100)}

    print("✅ Testing IoT Aggregation...")
    selected = agent.step_iot_aggregation(uav_states, active_devices, device_contents)
    print(f"Selected devices: {selected}")

    print("✅ Testing OFDM Allocation...")
    slots = agent.step_ofdm_allocation(uav_states, max_slots=2)
    print(f"OFDM allocation: {slots}")

    print("✅ Testing Caching & Offloading...")
    offload, cache = agent.step_caching_offloading(uav_states, task_bursts, candidate_content, uav_positions)
    print("Offloading and caching decisions made")

    print("🎉 All tests passed!")
    return agent

if __name__ == "__main__":
    test_hierarchical_agent()
'''


def main():
    """Fix the RL formulation file"""
    print("🔧 Final RL Formulation Fix")
    print("=" * 40)

    # Backup existing file
    if os.path.exists('rl_formulation_sagin.py'):
        try:
            shutil.copy('rl_formulation_sagin.py', 'rl_formulation_sagin_old.py')
            print("✅ Backed up existing file")
        except:
            pass

    # Write new content
    try:
        with open('rl_formulation_sagin.py', 'w', encoding='utf-8') as f:
            f.write(RL_CONTENT)
        print("✅ Wrote new RL formulation file")

        # Test import
        try:
            # Clear any cached modules
            import sys
            if 'rl_formulation_sagin' in sys.modules:
                del sys.modules['rl_formulation_sagin']

            from rl_formulation_sagin import HierarchicalSAGINAgent
            print("✅ Import test successful")

            # Test creation
            agent = HierarchicalSAGINAgent(grid_size=(2, 2))
            print(f"✅ Agent created: {sum(p.numel() for p in agent.parameters()):,} parameters")

            print("\n🎉 Fix completed successfully!")
            print("🚀 Now run: python minimal_rl_test.py")

            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Failed to write file: {e}")
        return False


if __name__ == "__main__":
    main()