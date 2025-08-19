# ofdm_allocation_baselines.py - OFDM Slot Allocation Baseline Approaches
import numpy as np
from collections import defaultdict


class BacklogGreedyOFDMAllocation:
    """
    Baseline 1: Backlog-greedy OFDM allocation
    Picks the S UAVs with largest aggregated data B_agg_u(t)
    """

    def __init__(self, total_ofdm_slots, num_satellites):
        self.total_ofdm_slots = total_ofdm_slots
        self.num_satellites = num_satellites

    def allocate_slots(self, uav_states):
        """
        Allocate OFDM slots based on aggregated data backlog

        Args:
            uav_states: {uav_coord: uav_state_dict} containing aggregated data info

        Returns:
            slot_allocation: {uav_coord: {sat_id: bool}} allocation decisions
            connected_uavs: set of UAVs with assigned subchannels
        """
        # Calculate backlog for each UAV
        uav_backlogs = []

        for uav_coord, uav_state in uav_states.items():
            # Calculate total aggregated data size
            aggregated_content = uav_state.get('aggregated_content', {})
            backlog = sum(content.get('size', 0) for content in aggregated_content.values())
            uav_backlogs.append((backlog, uav_coord))

        # Sort by backlog (descending)
        uav_backlogs.sort(key=lambda x: x[0], reverse=True)

        # Allocate slots to top S UAVs with highest backlog
        slot_allocation = {coord: {} for coord in uav_states.keys()}
        connected_uavs = set()

        slots_allocated = 0
        sat_assignments = defaultdict(list)  # Track assignments per satellite

        for backlog, uav_coord in uav_backlogs:
            if slots_allocated >= self.total_ofdm_slots:
                break

            if backlog <= 0:  # No data to upload
                continue

            # Find satellite with least assignments (load balancing)
            best_sat_id = min(range(self.num_satellites),
                              key=lambda s: len(sat_assignments[s]))

            # Assign slot
            slot_allocation[uav_coord][best_sat_id] = True
            sat_assignments[best_sat_id].append(uav_coord)
            connected_uavs.add(uav_coord)
            slots_allocated += 1

            print(f"Backlog-greedy: Assigned UAV {uav_coord} to Satellite {best_sat_id} "
                  f"(backlog: {backlog:.1f}MB)")

        print(f"Backlog-greedy allocation: {slots_allocated}/{self.total_ofdm_slots} slots used")
        return slot_allocation, connected_uavs


class RoundRobinOFDMAllocation:
    """
    Baseline 2: Round-robin OFDM allocation
    Cyclic slot assignment irrespective of backlog or queue
    """

    def __init__(self, total_ofdm_slots, num_satellites):
        self.total_ofdm_slots = total_ofdm_slots
        self.num_satellites = num_satellites
        self.round_robin_counter = 0  # Track current position in round-robin

    def allocate_slots(self, uav_states):
        """
        Allocate OFDM slots using round-robin scheduling
        """
        uav_coords = list(uav_states.keys())
        if not uav_coords:
            return {}, set()

        slot_allocation = {coord: {} for coord in uav_coords}
        connected_uavs = set()

        slots_allocated = 0
        sat_assignments = defaultdict(list)

        # Allocate slots in round-robin fashion
        for _ in range(self.total_ofdm_slots):
            if not uav_coords:
                break

            # Get next UAV in round-robin order
            uav_coord = uav_coords[self.round_robin_counter % len(uav_coords)]
            self.round_robin_counter += 1

            # Skip if this UAV already has a slot
            if uav_coord in connected_uavs:
                continue

            # Find satellite with least assignments
            best_sat_id = min(range(self.num_satellites),
                              key=lambda s: len(sat_assignments[s]))

            # Assign slot
            slot_allocation[uav_coord][best_sat_id] = True
            sat_assignments[best_sat_id].append(uav_coord)
            connected_uavs.add(uav_coord)
            slots_allocated += 1

            print(f"Round-robin: Assigned UAV {uav_coord} to Satellite {best_sat_id}")

        print(f"Round-robin allocation: {slots_allocated}/{self.total_ofdm_slots} slots used")
        return slot_allocation, connected_uavs


class RandomOFDMAllocation:
    """
    Baseline 3: Random OFDM allocation
    Choose S UAVs uniformly at random
    """

    def __init__(self, total_ofdm_slots, num_satellites):
        self.total_ofdm_slots = total_ofdm_slots
        self.num_satellites = num_satellites

    def allocate_slots(self, uav_states):
        """
        Allocate OFDM slots randomly
        """
        uav_coords = list(uav_states.keys())
        if not uav_coords:
            return {}, set()

        slot_allocation = {coord: {} for coord in uav_coords}
        connected_uavs = set()

        # Randomly select UAVs for slot allocation
        num_slots_to_allocate = min(self.total_ofdm_slots, len(uav_coords))
        selected_uavs = np.random.choice(
            uav_coords,
            size=num_slots_to_allocate,
            replace=False
        )

        sat_assignments = defaultdict(list)

        for i, uav_coord in enumerate(selected_uavs):
            # Assign to satellite in round-robin fashion among satellites
            sat_id = i % self.num_satellites

            slot_allocation[uav_coord][sat_id] = True
            sat_assignments[sat_id].append(uav_coord)
            connected_uavs.add(uav_coord)

            print(f"Random: Assigned UAV {uav_coord} to Satellite {sat_id}")

        print(f"Random allocation: {len(selected_uavs)}/{self.total_ofdm_slots} slots used")
        return slot_allocation, connected_uavs


class PriorityBasedOFDMAllocation:
    """
    Baseline 4: Priority-based OFDM allocation (current system approach)
    Combines aggregated data size and queue urgency for priority calculation
    """

    def __init__(self, total_ofdm_slots, num_satellites):
        self.total_ofdm_slots = total_ofdm_slots
        self.num_satellites = num_satellites

    def allocate_slots(self, uav_states):
        """
        Allocate OFDM slots based on priority (backlog + queue urgency)
        """
        # Calculate priority for each UAV
        uav_priorities = []

        for uav_coord, uav_state in uav_states.items():
            # Aggregated data size
            aggregated_content = uav_state.get('aggregated_content', {})
            aggregated_size = sum(content.get('size', 0) for content in aggregated_content.values())

            # Queue urgency (number of pending tasks)
            queue_length = len(uav_state.get('queue', []))

            # Combined priority (weight queue more heavily)
            priority = aggregated_size + queue_length * 2

            uav_priorities.append((priority, uav_coord, uav_state))

        # Sort by priority (descending)
        uav_priorities.sort(key=lambda x: x[0], reverse=True)

        slot_allocation = {coord: {} for coord in uav_states.keys()}
        connected_uavs = set()

        slots_allocated = 0
        sat_assignments = defaultdict(list)

        for priority, uav_coord, uav_state in uav_priorities:
            if slots_allocated >= self.total_ofdm_slots:
                break

            if priority <= 0:  # No urgent need for slot
                continue

            # Find satellite with least load
            best_sat_id = min(range(self.num_satellites),
                              key=lambda s: len(sat_assignments[s]))

            # Assign slot
            slot_allocation[uav_coord][best_sat_id] = True
            sat_assignments[best_sat_id].append(uav_coord)
            connected_uavs.add(uav_coord)
            slots_allocated += 1

            print(f"Priority-based: Assigned UAV {uav_coord} to Satellite {best_sat_id} "
                  f"(priority: {priority:.1f})")

        print(f"Priority-based allocation: {slots_allocated}/{self.total_ofdm_slots} slots used")
        return slot_allocation, connected_uavs


class FairnessAwareOFDMAllocation:
    """
    Baseline 5: Fairness-aware OFDM allocation
    Considers historical slot allocation to ensure fairness among UAVs
    """

    def __init__(self, total_ofdm_slots, num_satellites):
        self.total_ofdm_slots = total_ofdm_slots
        self.num_satellites = num_satellites
        self.allocation_history = defaultdict(int)  # Track historical allocations

    def allocate_slots(self, uav_states):
        """
        Allocate slots considering fairness (historical allocation balance)
        """
        # Calculate fairness-adjusted priority
        uav_priorities = []

        for uav_coord, uav_state in uav_states.items():
            # Current need (aggregated data + queue)
            aggregated_content = uav_state.get('aggregated_content', {})
            current_need = sum(content.get('size', 0) for content in aggregated_content.values())
            current_need += len(uav_state.get('queue', [])) * 2

            # Historical allocation count (lower = higher priority for fairness)
            historical_count = self.allocation_history[uav_coord]

            # Fairness-adjusted priority (higher current need, lower historical allocation = higher priority)
            if current_need > 0:
                priority = current_need / max(1, historical_count + 1)  # Add 1 to avoid division by zero
                uav_priorities.append((priority, uav_coord, uav_state))

        # Sort by priority (descending)
        uav_priorities.sort(key=lambda x: x[0], reverse=True)

        slot_allocation = {coord: {} for coord in uav_states.keys()}
        connected_uavs = set()

        slots_allocated = 0
        sat_assignments = defaultdict(list)

        for priority, uav_coord, uav_state in uav_priorities:
            if slots_allocated >= self.total_ofdm_slots:
                break

            # Find satellite with least load
            best_sat_id = min(range(self.num_satellites),
                              key=lambda s: len(sat_assignments[s]))

            # Assign slot
            slot_allocation[uav_coord][best_sat_id] = True
            sat_assignments[best_sat_id].append(uav_coord)
            connected_uavs.add(uav_coord)
            slots_allocated += 1

            # Update allocation history
            self.allocation_history[uav_coord] += 1

            print(f"Fairness-aware: Assigned UAV {uav_coord} to Satellite {best_sat_id} "
                  f"(priority: {priority:.2f}, history: {self.allocation_history[uav_coord]})")

        print(f"Fairness-aware allocation: {slots_allocated}/{self.total_ofdm_slots} slots used")
        return slot_allocation, connected_uavs


# Factory function
def create_ofdm_baseline(baseline_type, total_ofdm_slots, num_satellites, **kwargs):
    """
    Factory function to create OFDM allocation baselines

    Args:
        baseline_type: 'backlog_greedy', 'round_robin', 'random', 'priority_based', 'fairness_aware'
        total_ofdm_slots: Total number of available OFDM subchannels
        num_satellites: Number of satellites
        **kwargs: Additional arguments

    Returns:
        OFDM allocation baseline object
    """
    if baseline_type == 'backlog_greedy':
        return BacklogGreedyOFDMAllocation(total_ofdm_slots, num_satellites)
    elif baseline_type == 'round_robin':
        return RoundRobinOFDMAllocation(total_ofdm_slots, num_satellites)
    elif baseline_type == 'random':
        return RandomOFDMAllocation(total_ofdm_slots, num_satellites)
    elif baseline_type == 'priority_based':
        return PriorityBasedOFDMAllocation(total_ofdm_slots, num_satellites)
    elif baseline_type == 'fairness_aware':
        return FairnessAwareOFDMAllocation(total_ofdm_slots, num_satellites)
    else:
        raise ValueError(f"Unknown OFDM baseline type: {baseline_type}")


# Testing
if __name__ == "__main__":
    print("=== Testing OFDM Slot Allocation Baselines ===")

    # Mock UAV states for testing
    uav_states = {
        (0, 0): {
            'aggregated_content': {
                (0, 0, 1): {'size': 5.0},
                (0, 0, 2): {'size': 3.0}
            },
            'queue': [{'task_id': 1}, {'task_id': 2}]
        },
        (0, 1): {
            'aggregated_content': {
                (0, 1, 1): {'size': 8.0}
            },
            'queue': [{'task_id': 3}]
        },
        (1, 0): {
            'aggregated_content': {
                (1, 0, 1): {'size': 12.0},
                (1, 0, 2): {'size': 6.0}
            },
            'queue': []
        },
        (1, 1): {
            'aggregated_content': {},
            'queue': [{'task_id': 4}, {'task_id': 5}, {'task_id': 6}]
        }
    }

    total_slots = 3
    num_sats = 2

    # Test each baseline
    baseline_types = ['backlog_greedy', 'round_robin', 'random', 'priority_based', 'fairness_aware']

    for baseline_type in baseline_types:
        print(f"\n--- Testing {baseline_type} baseline ---")

        allocator = create_ofdm_baseline(baseline_type, total_slots, num_sats)
        slot_allocation, connected_uavs = allocator.allocate_slots(uav_states)

        # Verify constraint satisfaction
        total_assignments = sum(
            sum(assignments.values())
            for assignments in slot_allocation.values()
        )

        print(f"Total assignments: {total_assignments}/{total_slots}")
        print(f"Connected UAVs: {len(connected_uavs)}")
        print(f"Constraint satisfied: {total_assignments <= total_slots}")

        # Show assignments
        for uav_coord, sat_assignments in slot_allocation.items():
            for sat_id, assigned in sat_assignments.items():
                if assigned:
                    print(f"  UAV {uav_coord} -> Satellite {sat_id}")

    print("\n=== OFDM Slot Allocation Baselines Test Complete ===")