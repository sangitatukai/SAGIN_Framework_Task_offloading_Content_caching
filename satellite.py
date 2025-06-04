# satellite.py (MODIFIED FOR UNIVERSAL COVERAGE)
import numpy as np

class Satellite:
    def __init__(self, sat_id, storage_capacity, coverage_map, duration, position, compute_power):
        self.sat_id = sat_id
        self.storage = set()
        self.capacity = storage_capacity
        self.coverage_map = coverage_map
        self.current_coverage = set()
        self.position = position
        self.compute_power = compute_power
        self.global_content_set = set()
        self.fading_states = [1.0, 3.46, 5.03]
        self.current_channel_state = {}
        self.task_queue = []
        self.time_stamp = 0
        self.duration=duration
        self.tasks_within_bound = 0

    def receive_task(self, task, from_coord):
        # Calculate transmission and propagation delays
        source_pos = np.array(from_coord)
        target_pos = np.array(self.position)
        d = np.linalg.norm(source_pos - target_pos)
        bandwidth = 1e6
        propagation_speed = 3e8
        rate = bandwidth * np.log2(1 + 1 / (d ** 2 + 1e-9))
        task_size_bits = task.get('size', 2.0) * 8 * 1e6

        #transmission_delay = task_size_bits / rate
        propagation_delay = d / propagation_speed

        receive_time = task['generation_time'] + propagation_delay
        task['receive_time'] = receive_time
        #task['transmission_delay'] = transmission_delay
        task['propagation_delay'] = propagation_delay

        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t['receive_time'])
        print(f"[SAT {self.sat_id}] Queued task {task['task_id']} generated at {task['generation_time']} — will arrive at {receive_time:.2f}s")

    def update_coverage(self, timestep):
        # Assume full coverage of a 3x3 grid (adjust if different)
        self.current_coverage = {(x, y) for x in range(3) for y in range(3)}
        self.current_channel_state.clear()
        fading = self.fading_states[timestep % len(self.fading_states)]
        for (x, y) in self.current_coverage:
            self.current_channel_state[(x, y)] = fading

    def execute_tasks(self, timestep,  global_satellite_content_pool):
        completed = []
        next_available_time = timestep * self.duration
        for task in list(self.task_queue):
            cid = tuple(task['content_id'])
            content_info = global_satellite_content_pool.get(cid)
            if not content_info:
                print(f"[SAT {self.sat_id}] Skipped task {task['task_id']} — content not found")
                self.task_queue.remove(task)
                continue
            content_ready_time = content_info.get('receive_time_satellite', 0)
            if content_ready_time > task['receive_time']:
                print(
                    f"[SAT {self.sat_id}] Skipped task {task['task_id']} — content not ready by {task['receive_time']:.3f}s")
                self.task_queue.remove(task)
                continue

            start_time = max(task['receive_time'], next_available_time)
            compute_time = task['required_cpu'] / self.compute_power
            completion_time = start_time + compute_time
            total_delay = completion_time - task['generation_time']

            if total_delay > task['delay_bound']:
                print(f"[SAT {self.sat_id}] Dropped task {task['task_id']} — delay bound exceeded ({total_delay:.2f}s)")
                self.task_queue.remove(task)
                continue

            task['delay'] = {
                'queue_delay': start_time - task['receive_time'],
                'computation_delay': compute_time,
                'total_delay': total_delay,
                'execution_start': start_time,
                'completion_time': completion_time
            }

            next_available_time = completion_time
            completed.append(task)
            self.task_queue.remove(task)
            self.tasks_within_bound += 1
            print(f"[SAT {self.sat_id}] Executed task {task['task_id']} — done by {completion_time:.2f}s")
        self.next_available_time = next_available_time
        return completed

    def in_range(self, x, y):
        return True  # Universal coverage

    def get_channel_state(self, uav_coord):
        return self.current_channel_state.get(uav_coord, 1.0)  # Default to good channel if missing


    def has_content(self, content_id):
        return content_id in self.storage

    def compute_uplink_rate(self, transmit_power, channel_state, bandwidth):
        return bandwidth * np.log2(1 + channel_state * transmit_power)

