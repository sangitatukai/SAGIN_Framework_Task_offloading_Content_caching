# satellite.py - COMPLETE REPLACEMENT
import numpy as np
from communication_model import CommunicationModel


class Satellite:
    def __init__(self, sat_id, storage_capacity, coverage_map, duration, position, compute_power):
        # Basic satellite properties
        self.sat_id = sat_id
        self.position = position  # 3D position (x, y, altitude)
        self.storage_capacity = storage_capacity  # MB
        self.compute_power = compute_power  # CPU cycles per second
        self.duration = duration  # Time slot duration

        # Communication model
        self.comm_model = CommunicationModel()

        # Coverage and channel management
        self.coverage_map = coverage_map
        self.current_coverage = set()  # Currently covered regions
        self.channel_states = {}  # {uav_coord: fading_factor}
        self.fading_patterns = [1.0, 0.8, 1.2, 0.9, 1.1]  # Channel fading states

        # Content management
        self.local_storage = {}  # Local content storage {cid: metadata}
        self.global_content_set = set()  # Global content pool awareness G(t)
        self.storage_used_mb = 0.0  # Current storage usage

        # Task management
        self.task_queue = []  # Received tasks waiting for execution
        self.next_available_time = 0  # For sequential task execution

        # Performance metrics
        self.tasks_received = 0
        self.tasks_completed = 0
        self.tasks_within_bound = 0

        # Timing
        self.current_timestep = 0

        print(f"Satellite {sat_id} initialized at position {position} with {compute_power} compute power")

    def update_coverage(self, timestep):
        """
        Update satellite coverage and channel conditions
        Implements dynamic coverage patterns for LEO satellites
        """
        self.current_timestep = timestep

        # For simplicity, assume universal coverage (as per paper assumption)
        # In reality, LEO satellites have time-varying coverage
        self.current_coverage = {(x, y) for x in range(3) for y in range(3)}

        # Update channel fading states for each covered region
        self.channel_states = {}
        for coord in self.current_coverage:
            # Cycle through fading patterns with some randomness
            base_fading = self.fading_patterns[timestep % len(self.fading_patterns)]
            random_variation = np.random.uniform(0.8, 1.2)
            self.channel_states[coord] = base_fading * random_variation

        print(f"Satellite {self.sat_id}: Coverage updated for timestep {timestep}, "
              f"covering {len(self.current_coverage)} regions")

    def get_channel_state(self, uav_coord):
        """
        Get current channel fading state for UAV communication
        Used in Ka-band rate calculations
        """
        return self.channel_states.get(uav_coord, 1.0)

    def in_range(self, x, y):
        """Check if UAV coordinates are in satellite coverage"""
        return (x, y) in self.current_coverage

    def receive_task(self, task, from_coord):
        """
        Receive task from UAV with proper delay calculation
        Implements satellite task reception with propagation delays
        """
        self.tasks_received += 1

        # Calculate propagation and transmission delays
        if isinstance(from_coord, tuple) and len(from_coord) == 3:
            uav_pos = np.array(from_coord)
        else:
            # Convert UAV grid coordinates to 3D position
            x, y = from_coord
            uav_pos = np.array([x * 100 + 50, y * 100 + 50, 100])

        sat_pos = np.array(self.position)
        distance = np.linalg.norm(uav_pos - sat_pos)

        # Propagation delay (Paper Equation 10)
        propagation_speed = 3e8  # Speed of light
        propagation_delay = distance / propagation_speed

        # Transmission delay for task metadata (assume small size)
        task_metadata_bits = 1000  # 1KB for task metadata
        fading = self.get_channel_state(from_coord[:2] if len(from_coord) >= 2 else (0, 0))

        # Use Ka-band uplink rate for task metadata transmission
        rate, success, delay_func = self.comm_model.compute_uav_to_satellite_uplink_rate(
            uav_pos=uav_pos,
            sat_pos=sat_pos,
            subchannel_assigned=True,
            fading=fading
        )

        if success:
            transmission_delay = task_metadata_bits / rate
        else:
            transmission_delay = 0.001  # Minimal delay if rate calculation fails

        # Total reception delay
        total_reception_delay = propagation_delay + transmission_delay

        # Update task with satellite-specific timing
        task['receive_time'] = task['generation_time'] + total_reception_delay
        task['propagation_delay'] = propagation_delay
        task['transmission_delay'] = transmission_delay
        task['satellite_id'] = self.sat_id

        # Add to task queue (sorted by receive time)
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t['receive_time'])

        print(f"Satellite {self.sat_id}: Queued task {task['task_id']} from UAV, "
              f"receive_time={task['receive_time']:.3f}s, "
              f"delays=(prop={propagation_delay:.3f}s, trans={transmission_delay:.6f}s)")

        return True

    def execute_tasks(self, timestep, global_satellite_content_pool):
        """
        Execute queued tasks with proper computation model
        Implements Paper Equations 17-18 for satellite computing
        """
        completed = []
        current_time = max(timestep * self.duration,
                           getattr(self, 'next_available_time', timestep * self.duration))

        print(f"Satellite {self.sat_id}: Executing tasks at timestep {timestep}, "
              f"queue length: {len(self.task_queue)}")

        for task in list(self.task_queue):
            cid = tuple(task['content_id'])
            task_id = task['task_id']

            # Check if required content is available in global pool
            if cid not in global_satellite_content_pool:
                print(f"Satellite {self.sat_id}: Dropped task {task_id} - content {cid} not in global pool")
                self.task_queue.remove(task)
                continue

            content_info = global_satellite_content_pool[cid]
            content_available_time = content_info.get('received_by_satellite', 0)

            # Check if content is ready when task arrives
            if content_available_time > task['receive_time']:
                print(f"Satellite {self.sat_id}: Dropped task {task_id} - "
                      f"content not ready by task receive time "
                      f"({content_available_time:.3f}s > {task['receive_time']:.3f}s)")
                self.task_queue.remove(task)
                continue

            # Calculate execution timing
            execution_start_time = max(task['receive_time'], current_time)

            # Computation delay (Paper Equation 18)
            # τ^c(i)_k(t) = (1/f_k) × [Σ queued_tasks + Σ current_batch]
            computation_cycles = task['required_cpu']
            computation_delay = computation_cycles / self.compute_power
            completion_time = execution_start_time + computation_delay

            # Total delay from task generation to completion
            total_delay = completion_time - task['generation_time']

            # Check delay bound constraint (Paper Equation 28)
            if total_delay > task['delay_bound']:
                print(f"Satellite {self.sat_id}: Dropped task {task_id} - "
                      f"delay bound exceeded ({total_delay:.3f}s > {task['delay_bound']:.3f}s)")
                self.task_queue.remove(task)
                continue

            # Task execution successful
            task['delay'] = {
                'queue_delay': execution_start_time - task['receive_time'],
                'computation_delay': computation_delay,
                'propagation_delay': task.get('propagation_delay', 0),
                'transmission_delay': task.get('transmission_delay', 0),
                'total_delay': total_delay,
                'execution_start': execution_start_time,
                'completion_time': completion_time
            }

            # Update satellite state
            current_time = completion_time
            self.next_available_time = current_time
            completed.append(task)
            self.task_queue.remove(task)
            self.tasks_completed += 1
            self.tasks_within_bound += 1

            print(f"Satellite {self.sat_id}: Executed task {task_id} - "
                  f"completed at {completion_time:.3f}s, total_delay={total_delay:.3f}s")

        if completed:
            print(f"Satellite {self.sat_id}: Completed {len(completed)} tasks this timestep")

        return completed

    def store_content(self, content_id, content_metadata):
        """
        Store content in satellite local storage
        """
        if self.storage_used_mb + content_metadata['size'] <= self.storage_capacity:
            self.local_storage[content_id] = content_metadata
            self.storage_used_mb += content_metadata['size']
            return True
        else:
            print(f"Satellite {self.sat_id}: Storage full, cannot store {content_id}")
            return False

    def has_content(self, content_id):
        """Check if satellite has specific content locally or in global pool"""
        return content_id in self.local_storage or content_id in self.global_content_set

    def compute_uplink_rate(self, transmit_power, channel_state, bandwidth):
        """
        Compute uplink data rate (for compatibility)
        """
        # Simple rate calculation for backward compatibility
        snr = channel_state * transmit_power
        rate = bandwidth * np.log2(1 + snr)
        return rate

    def compute_downlink_rate_to_uav(self, uav_pos, channel_fading=1.0):
        """
        Compute downlink rate to specific UAV using proper Ka-band model
        """
        rate, success, delay_func = self.comm_model.compute_satellite_to_uav_downlink_rate(
            sat_pos=self.position,
            uav_pos=uav_pos,
            subchannel_assigned=True,
            fading=channel_fading
        )
        return rate, success

    def get_task_statistics(self):
        """Get current task execution statistics"""
        return {
            'satellite_id': self.sat_id,
            'tasks_received': self.tasks_received,
            'tasks_completed': self.tasks_completed,
            'tasks_within_bound': self.tasks_within_bound,
            'success_rate': (self.tasks_within_bound / self.tasks_received * 100) if self.tasks_received > 0 else 0,
            'current_queue_length': len(self.task_queue),
            'storage_usage': f"{self.storage_used_mb:.1f}/{self.storage_capacity}MB"
        }

    def get_coverage_statistics(self):
        """Get current coverage statistics"""
        return {
            'satellite_id': self.sat_id,
            'position': self.position,
            'current_coverage': list(self.current_coverage),
            'coverage_size': len(self.current_coverage),
            'channel_states': dict(self.channel_states)
        }

    def reset_performance_counters(self):
        """Reset performance counters for new simulation"""
        self.tasks_received = 0
        self.tasks_completed = 0
        self.tasks_within_bound = 0
        self.task_queue = []
        self.next_available_time = 0

    def estimate_execution_time(self, cpu_cycles):
        """Estimate execution time for given CPU cycles"""
        return cpu_cycles / self.compute_power

    def get_available_storage(self):
        """Get available storage space"""
        return self.storage_capacity - self.storage_used_mb

    def evict_content(self, content_ids):
        """Evict specified content from local storage"""
        for cid in content_ids:
            if cid in self.local_storage:
                self.storage_used_mb -= self.local_storage[cid]['size']
                del self.local_storage[cid]
                print(f"Satellite {self.sat_id}: Evicted content {cid}")

    def get_system_load(self):
        """Calculate current system load"""
        queue_load = len(self.task_queue) / 10  # Normalize by typical max queue
        storage_load = self.storage_used_mb / self.storage_capacity
        processing_load = min(1.0, len(self.task_queue) * 0.1)  # Rough processing load estimate

        return {
            'queue_load': queue_load,
            'storage_load': storage_load,
            'processing_load': processing_load,
            'overall_load': (queue_load + storage_load + processing_load) / 3
        }

    def can_accept_task(self, estimated_cpu_cycles, delay_bound):
        """
        Check if satellite can accept a new task within delay bounds
        """
        # Estimate current queue processing time
        queue_processing_time = sum(
            task['required_cpu'] / self.compute_power
            for task in self.task_queue
        )

        # Estimate this task's processing time
        task_processing_time = estimated_cpu_cycles / self.compute_power

        # Total estimated delay
        estimated_total_delay = queue_processing_time + task_processing_time

        return estimated_total_delay <= delay_bound

    def get_status_summary(self):
        """Get comprehensive satellite status"""
        task_stats = self.get_task_statistics()
        coverage_stats = self.get_coverage_statistics()
        load_stats = self.get_system_load()

        return {
            'satellite_id': self.sat_id,
            'position': self.position,
            'tasks': task_stats,
            'coverage': coverage_stats,
            'load': load_stats,
            'storage': f"{self.storage_used_mb:.1f}/{self.storage_capacity}MB",
            'queue_length': len(self.task_queue)
        }


# Testing and validation
if __name__ == "__main__":
    print("=== Testing Satellite Class ===")

    # Create test satellite
    satellite = Satellite(
        sat_id=0,
        storage_capacity=1000,
        coverage_map=[{(i, j) for i in range(3) for j in range(3)} for _ in range(5)],
        duration=300,
        position=(100, 500, 550000),
        compute_power=200
    )

    # Test coverage update
    satellite.update_coverage(timestep=0)
    print(f"Coverage: {satellite.get_coverage_statistics()}")

    # Test task reception
    test_task = {
        'task_id': 12345,
        'required_cpu': 50,
        'delay_bound': 5.0,
        'content_id': (1, 1, 5),
        'generation_time': 0.0,
        'size': 2.0
    }

    satellite.receive_task(test_task, from_coord=(1, 1, 100))

    # Test task execution (with mock global content pool)
    mock_global_pool = {
        (1, 1, 5): {
            'id': (1, 1, 5),
            'size': 2.0,
            'received_by_satellite': 0.1,
            'generation_time': 0.0
        }
    }

    completed = satellite.execute_tasks(timestep=0, global_satellite_content_pool=mock_global_pool)
    print(f"Executed {len(completed)} tasks")

    # Test statistics
    print(f"Task Statistics: {satellite.get_task_statistics()}")
    print(f"System Load: {satellite.get_system_load()}")
    print(f"Status Summary: {satellite.get_status_summary()}")

    # Test channel states
    for coord in [(0, 0), (1, 1), (2, 2)]:
        channel_state = satellite.get_channel_state(coord)
        print(f"Channel state for UAV {coord}: {channel_state:.3f}")

    print("\n=== Satellite Test Complete ===")