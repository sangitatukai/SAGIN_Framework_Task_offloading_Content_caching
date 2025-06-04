# uav.py (UPDATED: TTL-BASED EVICTION AND POPULARITY-BASED CACHING)
import numpy as np
from content_popularity_predictor import ContentPopularityPredictor

class UAV:
    def __init__(self, x, y, X, Y, cache_size, max_queue, duration, compute_power, energy, num_iot_per_region):
        self.pos = (x, y)
        self.energy = energy  # <--- energy should be a float or int, not a tuple!
        self.compute_power = compute_power
        self.cache_current = set()
        self.cache_storage = {}
        self.aggregated_content = []
        self.cache_capacity_mb = cache_size
        self.cache_used_mb = 0.0
        self.max_queue = max_queue

        self.uav_pos = (x * 100 + 50, y * 100 + 50, 100)
        self.energy_used_this_slot = 0.0
        self.neighbor_links = {}
        self.current_time = 0
        self.content_popularity = {}
        self.total_tasks = 0
        self.cache_hits = 0
        self.spatial_distribution = np.random.dirichlet(np.ones(X * Y))
        self.num_iot_per_region = num_iot_per_region
        self.duration = duration
        self.queue = []
        self.tasks_completed_within_bound = 0
        self.region = None  # ← We will set this later

    def update_cache(self, timestamp,  global_satellite_content_pool, is_connected_to_satellite):
        self.evict_expired_content(timestamp)

        candidate_pool = {}

        # 1. Add from local cache_storage (with all meta info)
        for cid, meta in self.cache_storage.items():
            candidate_pool[cid] = meta

        # 2. Add from aggregated_content (use all meta, update 'timestamp' for cache)
        for cid, content in self.aggregated_content.items():
            #meta = content.copy()
            #meta['timestamp'] = self.current_time  # Mark when it's available for cache
            candidate_pool[cid] = content

        # 3. Add from satellite global pool if connected (only if not present)
        if is_connected_to_satellite:
            for cid, sat_meta in global_satellite_content_pool.items():
                if cid not in candidate_pool:
                    meta = sat_meta.copy()
                    meta['timestamp'] = self.current_time
                    candidate_pool[cid] = meta

        # 4. Sort and select based on popularity
        popular, fallback = [], []
        for cid, meta in candidate_pool.items():
            if cid in self.content_popularity:
                popular.append((cid, self.content_popularity[cid], meta))
            else:
                fallback.append((cid, meta))

        popular_sorted = sorted(popular, key=lambda x: x[1], reverse=True)

        # 5. Build new cache within size limit
        new_cache = {}
        used_mb = 0.0
        for cid, _, meta in popular_sorted:
            size = meta['size']
            if used_mb + size <= self.cache_capacity_mb:
                new_cache[cid] = meta
                used_mb += size
        for cid, meta in fallback:
            if cid in new_cache:
                continue
            size = meta['size']
            if used_mb + size <= self.cache_capacity_mb:
                new_cache[cid] = meta
                used_mb += size

        self.cache_storage = new_cache
        self.cache_used_mb = used_mb

    def evict_expired_content(self, timestamp):
        expired = []
        for cid, meta in self.cache_storage.items():
            if ((timestamp+1)* self.duration) - meta['generation_time'] > meta['ttl']:
                expired.append(cid)
        for cid in expired:
            self.cache_used_mb -= self.cache_storage[cid]['size']
            del self.cache_storage[cid]
            self.cache_current.discard(cid)

    def receive_task(self, task, from_coord):
        cid = tuple(task['content_id'])

        if cid not in self.cache_storage and cid not in self.aggregated_content:
            print(f"[UAV {self.pos}] Dropped task {task['task_id']} — content not available")
            return

        self.content_popularity[cid] = self.content_popularity.get(cid, 0) + 1

        # Add 1–10ms jitter to receive time
        receive_jitter = np.random.uniform(0.001, 0.010)
        task['receive_time'] = task['generation_time'] + receive_jitter
        self.queue.append(task)
        self.queue.sort(key=lambda t: t['receive_time'])
        print(f"[UAV {self.pos}] Queued task {task['task_id']} at {task['receive_time']:.3f}s")

    def observe(self, neighbor_loads, satellite_in_range, activation_mask):
        obs = []
        obs.append(self.energy / 100.0)
        obs.append(len(self.queue) / self.max_queue)
        cache_vec = np.zeros(len(activation_mask))
        for content in self.cache_current:
            if content < len(cache_vec):
                cache_vec[content] = 1
        obs.extend(cache_vec)
        obs.extend(neighbor_loads.flatten())
        obs.extend(satellite_in_range)
        obs.extend(activation_mask)
        return np.array(obs, dtype=np.float32)

    def finalize_cache_update(self):
        pass

    def aggregate_content(self, content_dict):
        uav_pos = self.uav_pos
        time_so_far = 0
        aggregated = {}
        for cid, content in content_dict.items():
            iot_pos = content['iot_pos']
            data_rate, success = self.region.compute_iot_to_uav_rate(iot_pos, self.uav_pos)
            if not success or data_rate <= 0:
                continue
            content_bits = content['size'] * 8 * 1e6
            transmission_delay = content_bits / data_rate
            receive_time = content['generation_time'] + time_so_far + transmission_delay
            if receive_time - content['generation_time'] > self.duration:
                print(f"size, fail-{content['size']}")
                break
            content['received_by_uav'] = receive_time
            #print(f"content receive time: {content['received_by_uav']}")
            aggregated[cid] = content
            time_so_far += transmission_delay
            # energy deduction here as before

        # === Deduct energy: 1 per second of transmission (round up to ensure min 1 unit)
            energy_used = int(np.ceil(transmission_delay))
            self.energy -= energy_used
            self.energy_used_this_slot += energy_used

        self.aggregated_content = aggregated  # Always a dict!

    def clear_aggregated_content(self):
        """Clear aggregated IoT content after caching decision each slot."""
        self.aggregated_content = []

    # def update_cache(self, predicted_popularity):
    #     topk = np.argsort(predicted_popularity)[-self.cache_size:]
    #     self.cache_pending = set(topk)

    def finalize_cache_update(self):
        self.cache_current = self.cache_pending.copy()
        self.cache_pending.clear()

    def generate_tasks(self, X, Y, timestep, num_tasks=5):
        tasks = []
        x, y = self.pos
        base_time = timestep * self.duration  # duration = 300 seconds
        spatial_probs = self.spatial_distribution
        spatial_indices = [(i, j) for i in range(X) for j in range(Y)]

        num_tasks = np.random.randint(1, num_tasks + 1)
        offsets = sorted(np.random.choice(range(self.duration), size=num_tasks, replace=False))

        for offset in offsets:
            region_idx = np.random.choice(len(spatial_indices), p=spatial_probs)
            rand_x, rand_y = spatial_indices[region_idx]
            device_id = np.random.randint(0, self.num_iot_per_region)

            task = {
                'task_id': np.random.randint(10000),
                'required_cpu': np.random.randint(1, 5),
                'delay_bound': np.random.uniform(1.0, 10.0),
                'content_id': (rand_x, rand_y, device_id),
                'generation_time': base_time + offset,
                'remaining_cpu': None
            }
            tasks.append(task)

        return tasks

    def execute_tasks(self, timestep):
        print(f'im here {timestep}')
        completed = []
        current_time = getattr(self, 'next_available_time', timestep * 300)
        for task in list(self.queue):
            print(f'hi')
            cid = tuple(task['content_id'])
            content_ready_time = None

            # 1. If this is not the first slot, check cache storage (all content available at slot start)
            if timestep > 0 and cid in self.cache_storage:
                ready_time = timestep * self.duration
                if ready_time <= task['receive_time']:
                    content_ready_time = ready_time

            # 2. If not already set, check if content was freshly received this slot
            if content_ready_time is None and cid in self.aggregated_content:
                c = self.aggregated_content[cid]
                ready_time = c.get('received_by_uav', 0)
                if ready_time <= task['receive_time']:
                    content_ready_time = ready_time

            if content_ready_time is None:
                print(f"[UAV {self.pos}] Skipped task {task['task_id']} — content not ready by {task['receive_time']:.3f}s")
                self.queue.remove(task)
                continue

            # === Sequential task execution as before ===
            start_time = max(task['receive_time'], current_time)
            compute_time = task['required_cpu'] / self.compute_power
            completion_time = start_time + compute_time
            total_delay = completion_time - task['generation_time']
            print(f"strat time: {start_time}, completion_time: {completion_time}, total_delay: {total_delay}")
            if total_delay > task['delay_bound']:
                print(f"[UAV {self.pos}] Dropped task {task['task_id']} — delay bound exceeded ({total_delay:.2f}s)")

                self.queue.remove(task)
                continue

            task['delay'] = {
                'queue_delay': start_time - task['receive_time'],
                'computation_delay': compute_time,
                'total_delay': total_delay,
                'execution_start': start_time,
                'completion_time': completion_time
            }

            self.energy -= compute_time * 5
            self.energy_used_this_slot += compute_time * 5
            current_time = completion_time
            self.next_available_time = current_time
            self.cache_hits += int(task['content_id'] in self.cache_current)
            completed.append(task)
            self.queue.remove(task)
            self.tasks_completed_within_bound += 1
            print(f"[UAV {self.pos}] Executed task {task['task_id']} — done by {completion_time:.2f}s")
        return completed

    def update_content_popularity(self):
        # Optional: Decay old popularity to keep things adaptive
        for cid in list(self.content_popularity):
            self.content_popularity[cid] *= 0.9  # decay old values slightly







