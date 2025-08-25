# iot_aggregation_baselines.py - IoT Data Aggregation Baseline Approaches
import numpy as np
import torch
import torch.nn as nn
from communication_model import CommunicationModel


class PopularityToSizeGreedyAggregation:
    """
    Baseline 1: Popularity-to-size greedy aggregation
    Selects IoT devices based on popularity/size ratio within TDMA constraints
    """

    def __init__(self, duration=300):
        self.duration = duration
        self.comm_model = CommunicationModel()
        self.device_usefulness = {}  # Track device usefulness over time

    def select_devices(self, active_devices, content_dict, uav_pos, interfering_regions, slot_duration=None):
        """
        Select devices using popularity-to-size greedy approach

        Args:
            active_devices: List of active device IDs
            content_dict: {cid: content_metadata} for active devices
            uav_pos: UAV 3D position
            interfering_regions: List of interfering region coordinates
            slot_duration: TDMA slot duration (overrides self.duration if provided)

        Returns:
            selected_devices: List of selected device IDs
            aggregated_content: {cid: content_metadata} for selected devices
        """
        if not active_devices or not content_dict:
            return [], {}

        # Use provided slot_duration or default
        duration = slot_duration if slot_duration is not None else self.duration

        print(f"Greedy aggregation: {len(active_devices)} devices, {len(content_dict)} content items")

        # Calculate interference
        try:
            interference = self.comm_model.estimate_co_channel_interference(
                uav_pos, interfering_regions
            )
        except Exception:
            interference = 1e-12

        # Calculate priority scores for each device
        device_priorities = []

        for device_id in active_devices:
            # Find content for this device
            device_content = None
            for cid, content in content_dict.items():
                if content.get('device_id') == device_id:
                    device_content = content
                    break

            if device_content is None:
                continue

            # Validate required fields
            if not all(field in device_content for field in ['size', 'iot_pos']):
                print(f"Skipping device {device_id}: missing required fields")
                continue

            # Get usefulness score (popularity proxy)
            usefulness = self.device_usefulness.get(device_id, 1.0)

            # Calculate priority: usefulness / size (higher = better)
            priority = usefulness / max(device_content['size'], 0.1)  # Avoid division by zero

            device_priorities.append((priority, device_id, device_content))

        # Sort by priority (descending - highest priority first)
        device_priorities.sort(key=lambda x: x[0], reverse=True)

        # Greedily select devices within TDMA constraints
        selected_devices = []
        aggregated_content = {}
        total_time_used = 0

        for priority, device_id, content in device_priorities:
            try:
                # Check communication feasibility
                rate, success, delay_func = self.comm_model.compute_iot_to_uav_rate(
                    iot_pos=content.get('iot_pos', (0, 0, 0)),
                    uav_pos=uav_pos,
                    interference=interference
                )

                if not success:
                    continue

                # Calculate transmission time
                transmission_time = delay_func(content.get('size', 1.0))

                # Check TDMA constraint (Paper Constraint 24)
                if total_time_used + transmission_time > duration:
                    print(f"Greedy: TDMA capacity reached ({total_time_used:.2f}s/{duration}s)")
                    break

                # Select this device
                selected_devices.append(device_id)
                cid = tuple(content['id']) if 'id' in content else (0, 0, device_id)

                # Update content metadata
                content['received_by_uav'] = content.get('generation_time', 0) + total_time_used + transmission_time
                content['transmission_delay'] = transmission_time
                aggregated_content[cid] = content
                total_time_used += transmission_time

                print(f"Selected device {device_id}: priority={priority:.3f}, size={content.get('size', 0):.1f}MB")

            except Exception as e:
                print(f"Error processing device {device_id}: {e}")
                continue

        # Update usefulness scores based on selection
        for device_id in selected_devices:
            self.device_usefulness[device_id] = min(2.0, self.device_usefulness.get(device_id, 1.0) * 1.1)

        print(f"Greedy result: {len(selected_devices)}/{len(active_devices)} devices, "
              f"TDMA efficiency: {(total_time_used / duration) * 100:.1f}%")

        return selected_devices, aggregated_content

    def update_device_usefulness(self, device_id, was_useful):
        """Update usefulness score based on whether content was used"""
        current_score = self.device_usefulness.get(device_id, 1.0)
        # Exponential moving average
        self.device_usefulness[device_id] = 0.8 * int(was_useful) + 0.2 * current_score


class GRUContextualBanditAggregation(nn.Module):
    """
    Baseline 2: GRU contextual bandit aggregation
    Uses GRU for temporal patterns but no temporal credit assignment
    """

    def __init__(self, obs_dim=16, hidden_dim=32, duration=300):
        super().__init__()
        self.duration = duration
        self.comm_model = CommunicationModel()

        # GRU for temporal context
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.device_scorer = nn.Linear(hidden_dim + 4, 1)  # +4 for device features

        self.hidden_state = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def select_devices(self, active_devices, content_dict, uav_pos, interfering_regions,
                       temporal_context=None, slot_duration=None):
        """
        Select devices using GRU contextual bandit approach

        Args:
            active_devices: List of active device IDs
            content_dict: {cid: content_metadata} for active devices
            uav_pos: UAV 3D position
            interfering_regions: List of interfering region coordinates
            temporal_context: Previous observations for GRU
            slot_duration: TDMA slot duration (overrides self.duration if provided)

        Returns:
            selected_devices: List of selected device IDs
            aggregated_content: {cid: content_metadata} for selected devices
        """
        if not active_devices or not content_dict:
            return [], {}

        # Use provided slot_duration or default
        duration = slot_duration if slot_duration is not None else self.duration

        print(f"GRU bandit aggregation: {len(active_devices)} devices available")

        # Get temporal context from GRU
        if temporal_context is not None:
            try:
                context_tensor = torch.FloatTensor(temporal_context).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    gru_output, self.hidden_state = self.gru(context_tensor, self.hidden_state)
                    temporal_embedding = gru_output.squeeze()
            except Exception:
                temporal_embedding = torch.zeros(32)  # Fallback
        else:
            temporal_embedding = torch.zeros(32)  # Default embedding

        # Calculate interference
        try:
            interference = self.comm_model.estimate_co_channel_interference(
                uav_pos, interfering_regions
            )
        except Exception:
            interference = 1e-12

        # Score each device using GRU + contextual features
        device_scores = []

        for device_id in active_devices:
            # Find content for this device
            device_content = None
            for cid, content in content_dict.items():
                if content.get('device_id') == device_id:
                    device_content = content
                    break

            if device_content is None:
                continue

            # Validate required fields
            if not all(field in device_content for field in ['size']):
                continue

            try:
                # Create device feature vector
                device_features = torch.FloatTensor([
                    device_content.get('size', 1.0) / 20.0,  # Normalized size
                    device_content.get('ttl', 1200) / 1800.0,  # Normalized TTL
                    device_id / 20.0,  # Normalized device ID
                    1.0  # Bias term
                ])

                # Combine temporal and device features
                combined_features = torch.cat([temporal_embedding, device_features])

                # Get device score using neural network
                with torch.no_grad():
                    score = self.device_scorer(combined_features).item()

                device_scores.append((score, device_id, device_content))

            except Exception:
                continue

        # Sort by score (descending - highest score first)
        device_scores.sort(key=lambda x: x[0], reverse=True)

        # Greedily select devices within TDMA constraints
        selected_devices = []
        aggregated_content = {}
        total_time_used = 0

        for score, device_id, content in device_scores:
            try:
                # Check communication feasibility
                rate, success, delay_func = self.comm_model.compute_iot_to_uav_rate(
                    iot_pos=content.get('iot_pos', (0, 0, 0)),
                    uav_pos=uav_pos,
                    interference=interference
                )

                if not success:
                    continue

                # Calculate transmission time
                transmission_time = delay_func(content.get('size', 1.0))

                # Check TDMA constraint
                if total_time_used + transmission_time > duration:
                    print(f"GRU Bandit: TDMA capacity reached ({total_time_used:.2f}s/{duration}s)")
                    break

                # Select this device
                selected_devices.append(device_id)
                cid = tuple(content['id']) if 'id' in content else (0, 0, device_id)

                # Update content metadata
                content['received_by_uav'] = content.get('generation_time', 0) + total_time_used + transmission_time
                content['transmission_delay'] = transmission_time
                aggregated_content[cid] = content
                total_time_used += transmission_time

                print(f"Selected device {device_id}: GRU score={score:.3f}")

            except Exception as e:
                print(f"Error processing device {device_id}: {e}")
                continue

        print(f"GRU Bandit result: {len(selected_devices)}/{len(active_devices)} devices")

        return selected_devices, aggregated_content

    def update_with_feedback(self, device_id, reward):
        """
        Update model based on immediate feedback (contextual bandit style)
        No temporal credit assignment - just immediate reward
        """
        try:
            # Simple loss based on immediate reward
            if hasattr(self, 'last_prediction'):
                loss = -reward * self.last_prediction  # Bandit-style loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        except Exception:
            pass  # Ignore update errors


class RandomAggregation:
    """
    Baseline 3: Random aggregation for comparison
    Randomly selects devices within TDMA constraints
    """

    def __init__(self, duration=300):
        self.duration = duration
        self.comm_model = CommunicationModel()

    def select_devices(self, active_devices, content_dict, uav_pos, interfering_regions, slot_duration=None):
        """
        Random device selection within TDMA constraints

        Args:
            active_devices: List of active device IDs
            content_dict: {cid: content_metadata} for active devices
            uav_pos: UAV 3D position
            interfering_regions: List of interfering region coordinates
            slot_duration: TDMA slot duration (overrides self.duration if provided)

        Returns:
            selected_devices: List of selected device IDs
            aggregated_content: {cid: content_metadata} for selected devices
        """
        if not active_devices or not content_dict:
            return [], {}

        # Use provided slot_duration or default
        duration = slot_duration if slot_duration is not None else self.duration

        print(f"Random aggregation: {len(active_devices)} devices available")

        # Calculate interference
        try:
            interference = self.comm_model.estimate_co_channel_interference(
                uav_pos, interfering_regions
            )
        except Exception:
            interference = 1e-12

        # Shuffle devices randomly
        shuffled_devices = active_devices.copy()
        np.random.shuffle(shuffled_devices)

        # Select devices within TDMA constraints
        selected_devices = []
        aggregated_content = {}
        total_time_used = 0

        for device_id in shuffled_devices:
            # Find content for this device
            device_content = None
            for cid, content in content_dict.items():
                if content.get('device_id') == device_id:
                    device_content = content
                    break

            if device_content is None:
                continue

            # Validate required fields
            if not all(field in device_content for field in ['size']):
                continue

            try:
                # Check communication feasibility
                rate, success, delay_func = self.comm_model.compute_iot_to_uav_rate(
                    iot_pos=device_content.get('iot_pos', (0, 0, 0)),
                    uav_pos=uav_pos,
                    interference=interference
                )

                if not success:
                    continue

                # Calculate transmission time
                transmission_time = delay_func(device_content.get('size', 1.0))

                # Check TDMA constraint
                if total_time_used + transmission_time > duration:
                    print(f"Random: TDMA capacity reached ({total_time_used:.2f}s/{duration}s)")
                    break

                # Select this device
                selected_devices.append(device_id)
                cid = tuple(device_content['id']) if 'id' in device_content else (0, 0, device_id)

                # Update content metadata
                device_content['received_by_uav'] = device_content.get('generation_time',
                                                                       0) + total_time_used + transmission_time
                device_content['transmission_delay'] = transmission_time
                aggregated_content[cid] = device_content
                total_time_used += transmission_time

            except Exception as e:
                print(f"Error processing device {device_id}: {e}")
                continue

        print(f"Random result: {len(selected_devices)}/{len(active_devices)} devices")

        return selected_devices, aggregated_content


# Factory function to create aggregation baselines
def create_aggregation_baseline(baseline_type, **kwargs):
    """
    Factory function to create IoT aggregation baselines

    Args:
        baseline_type: 'greedy', 'gru_bandit', or 'random'
        **kwargs: Additional arguments for baseline constructors

    Returns:
        Baseline aggregation object
    """
    if baseline_type == 'greedy':
        return PopularityToSizeGreedyAggregation(**kwargs)
    elif baseline_type == 'gru_bandit':
        return GRUContextualBanditAggregation(**kwargs)
    elif baseline_type == 'random':
        return RandomAggregation(**kwargs)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


# Testing and validation
if __name__ == "__main__":
    print("=== Testing IoT Aggregation Baselines ===")

    # Mock data for testing
    active_devices = [0, 1, 2, 3, 4]
    content_dict = {}
    for i in active_devices:
        content_dict[(1, 1, i)] = {
            'id': (1, 1, i),
            'size': np.random.uniform(2, 10),
            'ttl': 1200,
            'generation_time': 0,
            'iot_pos': (110 + i * 10, 110, 0),
            'device_id': i
        }

    uav_pos = (150, 150, 100)
    interfering_regions = [(0, 0), (2, 2)]

    # Test each baseline
    baselines = ['greedy', 'gru_bandit', 'random']

    for baseline_type in baselines:
        print(f"\n--- Testing {baseline_type} baseline ---")

        try:
            aggregator = create_aggregation_baseline(baseline_type)

            if baseline_type == 'gru_bandit':
                temporal_context = np.random.randn(16)
                selected, aggregated = aggregator.select_devices(
                    active_devices, content_dict, uav_pos, interfering_regions,
                    temporal_context=temporal_context, slot_duration=300
                )
            else:
                selected, aggregated = aggregator.select_devices(
                    active_devices, content_dict, uav_pos, interfering_regions,
                    slot_duration=300
                )

            print(f"Selected devices: {selected}")
            print(f"Aggregated content: {len(aggregated)} items")
            total_size = sum(content.get('size', 0) for content in aggregated.values())
            print(f"Total content size: {total_size:.1f}MB")

        except Exception as e:
            print(f"Error testing {baseline_type}: {e}")

    print("\n=== IoT Aggregation Baselines Test Complete ===")