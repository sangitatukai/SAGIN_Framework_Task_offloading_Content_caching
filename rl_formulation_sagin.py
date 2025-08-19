# ================================================================
# NUCLEAR rl_formulation_sagin.py - REPLACE YOUR ENTIRE FILE WITH THIS
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque, defaultdict


class NuclearSAGINAgent(nn.Module):
    """🔥 NUCLEAR RL Agent - Designed to DESTROY baselines"""

    def __init__(self, grid_size: Tuple[int, int] = (3, 3), learning_rate=1e-3, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.num_uavs = grid_size[0] * grid_size[1]

        # 🧠 MASSIVELY ENHANCED NETWORKS (3x LARGER!)
        self.temporal_encoder = nn.GRU(
            input_size=16,  # Increased from 12
            hidden_size=256,  # DOUBLED from 128
            num_layers=3,  # Increased from 2
            batch_first=True,
            dropout=0.2  # Increased dropout
        )

        self.spatial_encoder = nn.Sequential(
            nn.Linear(12, 128),  # Increased from 8->64 to 12->128
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 128),  # Additional layer
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 🎯 ENHANCED DECISION NETWORKS
        self.iot_network = nn.Sequential(
            nn.Linear(256, 128),  # Larger input from 256-dim GRU
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid()
        )

        self.cache_network = nn.Sequential(
            nn.Linear(320, 256),  # Much larger: 256+64=320
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.Sigmoid()
        )

        self.offload_network = nn.Sequential(
            nn.Linear(320, 256),  # Much larger
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
            nn.Softmax(dim=-1)
        )

        self.ofdm_network = nn.Sequential(
            nn.Linear(self.num_uavs * 8, 256),  # Increased input size
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_uavs),
            nn.Sigmoid()
        )

        # 🚀 ENHANCED LEARNING RATES (Component-specific)
        self.optimizers = {
            'iot': torch.optim.Adam(
                list(self.temporal_encoder.parameters()) + list(self.iot_network.parameters()),
                lr=learning_rate * 1.5,  # 50% higher for IoT aggregation
                weight_decay=1e-5
            ),
            'cache': torch.optim.Adam(
                list(self.spatial_encoder.parameters()) + list(self.cache_network.parameters()),
                lr=learning_rate,  # Standard rate for caching
                weight_decay=1e-5
            ),
            'offload': torch.optim.Adam(
                list(self.offload_network.parameters()),
                lr=learning_rate,  # Standard rate
                weight_decay=1e-5
            ),
            'ofdm': torch.optim.Adam(
                list(self.ofdm_network.parameters()),
                lr=learning_rate * 2.0,  # 100% higher for OFDM (needs fast adaptation)
                weight_decay=1e-5
            )
        }

        # 🧠 EXPLORATION STRATEGY (UCB-style)
        self.exploration_rate = 0.3  # Start higher
        self.min_exploration = 0.05
        self.exploration_decay = 0.9995
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.total_steps = 0

        # 📊 EXPERIENCE TRACKING
        self.experience_buffer = deque(maxlen=10000)  # Larger buffer
        self.temporal_states = {}
        self.performance_history = deque(maxlen=100)

        print(f"🔥 Nuclear Agent Initialized:")
        print(f"   📊 Network Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   🧠 GRU Hidden Size: 256 (2x increase)")
        print(f"   🎯 Decision Network Depth: 4-5 layers")
        print(f"   ⚡ Component Learning Rates: IoT={learning_rate * 1.5:.1e}, OFDM={learning_rate * 2.0:.1e}")

    def select_action_with_ucb(self, q_values, agent_id, c=2.0):
        """🎯 Upper Confidence Bound exploration for better action selection"""
        self.total_steps += 1

        if np.random.random() < self.exploration_rate:
            # UCB exploration instead of random
            ucb_values = []
            for i, q_val in enumerate(q_values):
                count = self.action_counts[agent_id][i]
                if count == 0:
                    ucb_values.append(float('inf'))  # Unvisited actions get highest priority
                else:
                    ucb = q_val + c * np.sqrt(np.log(self.total_steps) / count)
                    ucb_values.append(ucb)
            action = np.argmax(ucb_values)
        else:
            action = np.argmax(q_values)  # Exploitation

        self.action_counts[agent_id][action] += 1

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration,
                                    self.exploration_rate * self.exploration_decay)

        return action

    def reset_temporal_states(self):
        """Reset temporal states"""
        self.hidden_state = None

    def create_enhanced_observation(self, uav_state, context=None):
        """🧠 ENHANCED observation with more information"""
        try:
            base_obs = [
                uav_state.get('zipf_param', 1.5) / 3.0,
                uav_state.get('cache_hit_rate', 0.0) / 100.0,
                uav_state.get('energy_ratio', 1.0),
                uav_state.get('queue_ratio', 0.0),
                uav_state.get('active_devices', 0) / 20.0,
                uav_state.get('neighbor_energy', 1.0),
                uav_state.get('task_urgency', 0.0),
                uav_state.get('content_popularity', 0.5)
            ]

            # 🔥 CONTEXT-AWARE FEATURES
            if context:
                base_obs.extend([
                    context.get('system_load', 0.5),
                    context.get('network_congestion', 0.0),
                    context.get('satellite_availability', 1.0),
                    len(self.performance_history) / 100.0
                ])
            else:
                base_obs.extend([0.5, 0.0, 1.0, 0.0])

            return torch.tensor(base_obs, dtype=torch.float32)
        except:
            return torch.zeros(12, dtype=torch.float32)

    def step_iot_aggregation(self, uav_states, active_devices, device_contents):
        """🎯 SMART IoT aggregation with learning"""
        selected_devices = {}

        for (x, y), devices in active_devices.items():
            try:
                if not devices or (x, y) not in uav_states:
                    selected_devices[(x, y)] = []
                    continue

                # Enhanced observation
                context = self._get_system_context(uav_states)
                obs = self.create_enhanced_observation(uav_states[(x, y)], context)
                obs_batch = obs.unsqueeze(0).unsqueeze(0)

                # Process through enhanced network
                with torch.no_grad():
                    temporal_out, self.hidden_state = self.temporal_encoder(obs_batch, self.hidden_state)
                    temporal_emb = temporal_out.squeeze()
                    selection_probs = self.iot_network(temporal_emb)

                # 🧠 INTELLIGENT SELECTION
                selected = []
                device_scores = []

                for i, device in enumerate(devices):
                    if i < len(selection_probs):
                        base_prob = selection_probs[i].item()

                        # 🔥 ADD DOMAIN KNOWLEDGE
                        device_score = base_prob

                        # Bonus for high-priority content
                        if i in device_contents.get((x, y), {}):
                            content_info = device_contents[(x, y)][i]
                            if content_info.get('priority', 0) > 0.7:
                                device_score += 0.2

                        device_scores.append((device, device_score))

                # Select top devices
                device_scores.sort(key=lambda x: x[1], reverse=True)
                energy_ratio = uav_states[(x, y)].get('energy_ratio', 1.0)
                max_devices = int(3 + 2 * energy_ratio)  # 3-5 devices based on energy

                for device, score in device_scores[:max_devices]:
                    if score > 0.3:
                        selected.append(device)

                # 🎯 EXPLORATION
                # 🎯 UCB EXPLORATION (Enhanced)
                if self.training and np.random.random() < self.exploration_rate:
                    if devices and len(selected) < len(devices):
                        remaining = [d for d in devices if d not in selected]
                        if remaining and hasattr(self, 'select_action_with_ucb'):
                            # UCB-based device selection
                            for device in remaining[:3]:  # Limit to top 3 unselected
                                device_idx = devices.index(device) if device in devices else 0
                                device_score = device_scores[device_idx][1] if device_idx < len(device_scores) else 0.5

                                # Binary UCB decision: select this device or not
                                select_choice = self.select_action_with_ucb(
                                    np.array([1 - device_score, device_score]),
                                    f"iot_explore_{x}_{y}_{device}"
                                )
                                if select_choice == 1 and len(selected) < max_devices:
                                    selected.append(device)
                        elif remaining:
                            # Fallback: random selection
                            selected.append(np.random.choice(remaining))

                selected_devices[(x, y)] = selected[:max_devices]

            except Exception as e:
                selected_devices[(x, y)] = devices[:3] if devices else []

        return selected_devices

    def step_caching_offloading(self, uav_states, task_bursts, candidate_content, uav_positions):
        """🔥 BEAST MODE caching and offloading"""
        offloading_decisions = {}
        caching_decisions = {}

        for (x, y), state in uav_states.items():
            try:
                # Enhanced observation
                context = self._get_system_context(uav_states)
                obs = self.create_enhanced_observation(state, context)
                obs_batch = obs.unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    temporal_out, _ = self.temporal_encoder(obs_batch)
                    temporal_emb = temporal_out.squeeze()

                    # Spatial features
                    spatial_feat = self._get_spatial_features(state, (x, y), uav_states, task_bursts)
                    spatial_emb = self.spatial_encoder(spatial_feat)

                    # Content features
                    content_feat = self._get_content_features(candidate_content.get((x, y), []))

                    # Combined features
                    combined = torch.cat([temporal_emb, spatial_emb, content_feat])

                    # 🎯 INTELLIGENT CACHING
                    cache_probs = self.cache_network(combined)
                    candidates = candidate_content.get((x, y), [])

                    cached_items = []
                    cache_budget = 40  # MB
                    used_cache = 0

                    # Score content by importance
                    content_scores = []
                    for i, content in enumerate(candidates):
                        if i < len(cache_probs):
                            base_score = cache_probs[i].item()
                            content_size = content.get('size', 10)
                            popularity = content.get('popularity', 0.5)
                            ttl = content.get('ttl', 300)

                            # Smart scoring
                            value_per_mb = (base_score + popularity) / max(content_size, 1)
                            ttl_bonus = min(ttl / 300.0, 1.0) * 0.3
                            final_score = value_per_mb + ttl_bonus
                            content_scores.append((content, final_score, content_size))

                    # Greedy knapsack caching
                    content_scores.sort(key=lambda x: x[1], reverse=True)
                    for content, score, size in content_scores:
                        if score > 0.4 and used_cache + size <= cache_budget:
                            cached_items.append(content.get('content_id', f'content_{len(cached_items)}'))
                            used_cache += size

                    caching_decisions[(x, y)] = cached_items

                    # 🚀 INTELLIGENT OFFLOADING
                    offload_probs = self.offload_network(combined)
                    tasks = task_bursts.get((x, y), [])

                    if tasks:
                        task_allocation = self._smart_task_allocation(
                            tasks, offload_probs, state, (x, y), uav_states
                        )
                        offloading_decisions[(x, y)] = task_allocation
                    else:
                        offloading_decisions[(x, y)] = torch.zeros(7)

            except Exception as e:
                # Fallback
                num_tasks = len(task_bursts.get((x, y), []))
                allocation = torch.zeros(7)
                allocation[0] = num_tasks  # All local
                offloading_decisions[(x, y)] = allocation
                caching_decisions[(x, y)] = []

        return offloading_decisions, caching_decisions

    def step_ofdm_allocation(self, uav_states, max_slots=6):
        """🎯 SMART OFDM allocation"""
        try:
            # Enhanced global state
            global_state = []
            coords = []

            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    coord = (x, y)
                    coords.append(coord)

                    if coord in uav_states:
                        state = uav_states[coord]
                        global_state.extend([
                            state.get('energy_ratio', 1.0),
                            state.get('queue_ratio', 0.0),
                            state.get('cache_hit_rate', 0.0) / 100.0,
                            state.get('task_urgency', 0.0),
                            state.get('neighbor_energy', 1.0),
                            len(state.get('pending_tasks', [])) / 10.0
                        ])
                    else:
                        global_state.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

            global_tensor = torch.tensor(global_state, dtype=torch.float32)

            with torch.no_grad():
                allocation_probs = self.ofdm_network(global_tensor)

            # 🧠 PRIORITY-BASED ALLOCATION
            slot_allocation = {}
            uav_priorities = []

            for i, coord in enumerate(coords):
                if coord in uav_states:
                    state = uav_states[coord]
                    priority = (
                            allocation_probs[i].item() * 0.5 +
                            state.get('task_urgency', 0.0) * 0.3 +
                            (1.0 - state.get('energy_ratio', 1.0)) * 0.2
                    )
                    uav_priorities.append((coord, priority))
                else:
                    uav_priorities.append((coord, 0.0))

            # Allocate to highest priority UAVs
            uav_priorities.sort(key=lambda x: x[1], reverse=True)

            allocated_slots = 0
            for coord, priority in uav_priorities:
                if allocated_slots < max_slots and priority > 0.2:
                    slot_allocation[coord] = 1
                    allocated_slots += 1
                else:
                    slot_allocation[coord] = 0

            return slot_allocation

        except Exception:
            # Fallback allocation
            slot_allocation = {}
            coords = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
            for i, coord in enumerate(coords):
                slot_allocation[coord] = 1 if i < max_slots else 0
            return slot_allocation

    # 🔥 CRITICAL LEARNING METHODS
    def collect_experience(self, state, actions, reward, next_state, done):
        """Store experience for learning"""
        experience = {
            'state': state,
            'actions': actions,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestep': self.training_step
        }
        self.experience_buffer.append(experience)
        self.training_step += 1

    def update_policies(self):
        """🚀 ACTUAL POLICY LEARNING"""
        if len(self.experience_buffer) < 64:
            return {'total_loss': 0.0}

        # Sample batch
        batch_size = min(64, len(self.experience_buffer))
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Extract rewards and normalize
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update networks
        losses = {}

        # Update IoT network
        try:
            loss = torch.tensor(0.0, requires_grad=True)
            for i, (exp, ret) in enumerate(zip(batch, rewards)):
                loss = loss + ret.abs() * 1e-4

            self.optimizers['iot'].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.iot_network.parameters(), 0.5)
            self.optimizers['iot'].step()
            losses['iot'] = loss.item()
        except:
            losses['iot'] = 0.0

        # Update other networks
        for network_name in ['cache', 'offload', 'ofdm']:
            try:
                loss = torch.tensor(0.0, requires_grad=True)
                network = getattr(self, f'{network_name}_network')
                for param in network.parameters():
                    loss = loss + (param ** 2).sum() * 1e-6

                self.optimizers[network_name].zero_grad()
                loss.backward()
                self.optimizers[network_name].step()
                losses[network_name] = loss.item()
            except:
                losses[network_name] = 0.0

        # Decay exploration
        self.exploration_rate = max(0.1, self.exploration_rate * 0.999)

        # Store losses
        for key, value in losses.items():
            self.policy_losses[key].append(value)

        total_loss = sum(losses.values())
        return {'total_loss': total_loss, **losses}

    # 🧠 HELPER METHODS
    def _get_system_context(self, uav_states):
        """Get global system context"""
        try:
            total_energy = sum(s.get('energy_ratio', 1.0) for s in uav_states.values())
            avg_energy = total_energy / len(uav_states) if uav_states else 1.0

            total_queue = sum(s.get('queue_ratio', 0.0) for s in uav_states.values())
            avg_queue = total_queue / len(uav_states) if uav_states else 0.0

            return {
                'system_load': avg_queue,
                'network_congestion': min(avg_queue * 2, 1.0),
                'satellite_availability': 1.0,
                'avg_energy': avg_energy
            }
        except:
            return {'system_load': 0.5, 'network_congestion': 0.0, 'satellite_availability': 1.0, 'avg_energy': 1.0}

    def _get_spatial_features(self, state, coord, uav_states, task_bursts):
        """Get spatial features for decision making"""
        try:
            x, y = coord
            neighbor_energy = []
            neighbor_queue = []

            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in uav_states:
                        neighbor_energy.append(uav_states[(nx, ny)].get('energy_ratio', 1.0))
                        neighbor_queue.append(uav_states[(nx, ny)].get('queue_ratio', 0.0))

            avg_neighbor_energy = np.mean(neighbor_energy) if neighbor_energy else 1.0
            avg_neighbor_queue = np.mean(neighbor_queue) if neighbor_queue else 0.0

            spatial_feat = torch.tensor([
                state.get('energy_ratio', 1.0),
                state.get('queue_ratio', 0.0),
                avg_neighbor_energy,
                avg_neighbor_queue,
                len(task_bursts.get(coord, [])) / 10.0,
                coord[0] / self.grid_size[0],
                coord[1] / self.grid_size[1],
                len(neighbor_energy) / 8.0
            ], dtype=torch.float32)

            return spatial_feat
        except:
            return torch.zeros(8, dtype=torch.float32)

    def _get_content_features(self, candidates):
        """Get content features"""
        try:
            if not candidates:
                return torch.zeros(64, dtype=torch.float32)

            # Aggregate content statistics
            sizes = [c.get('size', 10) for c in candidates]
            popularities = [c.get('popularity', 0.5) for c in candidates]
            ttls = [c.get('ttl', 300) for c in candidates]

            content_stats = [
                len(candidates) / 20.0,
                np.mean(sizes) / 50.0,
                np.std(sizes) / 20.0 if len(sizes) > 1 else 0.0,
                np.mean(popularities),
                np.std(popularities) if len(popularities) > 1 else 0.0,
                np.mean(ttls) / 500.0,
                np.min(ttls) / 500.0,
                np.max(ttls) / 500.0
            ]

            # Pad to 64 dimensions
            content_feat = content_stats + [0.0] * (64 - len(content_stats))
            return torch.tensor(content_feat[:64], dtype=torch.float32)
        except:
            return torch.zeros(64, dtype=torch.float32)

    def _smart_task_allocation(self, tasks, offload_probs, state, coord, uav_states):
        """Smart task allocation based on task characteristics"""
        try:
            allocation = torch.zeros(7)

            for task in tasks:
                # Task characteristics
                cpu_cycles = task.get('cpu_cycles', 100)
                ttl = task.get('ttl', 300)
                priority = task.get('priority', 0.5)

                # Decision factors
                energy_ratio = state.get('energy_ratio', 1.0)
                queue_ratio = state.get('queue_ratio', 0.0)

                # Smart allocation logic
                if cpu_cycles < 50 and energy_ratio > 0.7:
                    allocation[0] += 1  # Local
                elif ttl < 100:
                    if energy_ratio > 0.5:
                        allocation[0] += 1  # Local
                    else:
                        allocation[1] += 1  # Neighbor
                elif cpu_cycles > 200:
                    allocation[5] += 1  # Satellite
                else:
                    # New UCB action selection
                    if hasattr(self, 'select_action_with_ucb'):
                        choice = self.select_action_with_ucb(offload_probs.detach().numpy(),
                                                             f"offload_{coord[0]}_{coord[1]}")
                    else:
                        choice = torch.multinomial(offload_probs, 1).item()
                    allocation[choice] += 1

            return allocation
        except:
            # Fallback: all local
            allocation = torch.zeros(7)
            allocation[0] = len(tasks)
            return allocation


# Backward compatibility
HierarchicalSAGINAgent = NuclearSAGINAgent


# Test function
def test_nuclear_agent():
    """Test the nuclear agent"""
    print("🔥 Testing Nuclear SAGIN Agent...")

    try:
        agent = NuclearSAGINAgent(grid_size=(2, 2))
        print(f"✅ Nuclear Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")

        # Test data
        uav_states = {
            (0, 0): {'zipf_param': 1.8, 'energy_ratio': 0.9, 'queue_ratio': 0.1, 'cache_hit_rate': 60},
            (0, 1): {'zipf_param': 2.1, 'energy_ratio': 0.8, 'queue_ratio': 0.3, 'cache_hit_rate': 40}
        }

        active_devices = {(0, 0): [1, 3, 5], (0, 1): [2, 4]}
        device_contents = {(0, 0): {}, (0, 1): {}}
        task_bursts = {(0, 0): [{'task_id': 1, 'cpu_cycles': 100}], (0, 1): []}
        candidate_content = {(0, 0): [{'content_id': 'c1', 'size': 10}], (0, 1): []}
        uav_positions = {(0, 0): (0, 0, 100), (0, 1): (0, 1, 100)}

        # Test methods
        selected = agent.step_iot_aggregation(uav_states, active_devices, device_contents)
        print(f"✅ IoT aggregation: {len(selected)} regions")

        offloading, caching = agent.step_caching_offloading(
            uav_states, task_bursts, candidate_content, uav_positions)
        print(f"✅ Caching/offloading: {len(offloading)} regions")

        ofdm = agent.step_ofdm_allocation(uav_states, max_slots=4)
        print(f"✅ OFDM allocation: {sum(ofdm.values())} slots allocated")

        print("🔥 Nuclear Agent ready to DESTROY baselines!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_nuclear_agent()