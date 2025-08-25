# uav.py - COMPLETE REPLACEMENT
import numpy as np
from communication_model import CommunicationModel


class UAV:
    def __init__(self, x, y, X, Y, cache_size, max_queue, duration, compute_power, energy, num_iot_per_region):
        # Basic UAV properties
        self.pos = (x, y)
        self.uav_pos = (x * 100 + 50, y * 100 + 50, 100)  # 3D position: region center + 100m altitude
        self.X, self.Y = X, Y  # Grid dimensions

        # Resource constraints
        self.energy = energy  # Initial energy (Joules)
        self.max_energy = energy  # Store initial energy for calculations
        self.compute_power = compute_power  # CPU cycles per second
        self.cache_capacity_mb = cache_size  # Cache capacity in MB
        self.max_queue = max_queue  # Maximum task queue length
        self.duration = duration  # Time slot duration (seconds)

        # Cache management
        self.cache_storage = {}  # Current cache: {content_id: metadata}
        self.cache_used_mb = 0.0  # Current cache usage
        self.aggregated_content = {}  # Content collected this slot: {content_id: metadata}
        self.content_popularity = {}  # Content popularity tracking: {content_id: score}

        # Task management
        self.queue = []  # Task queue
        self.next_available_time = 0  # For sequential task execution

        # Communication model
        self.comm_model = CommunicationModel()

        # Energy tracking
        self.energy_used_this_slot = 0.0

        # Performance metrics
        self.total_tasks = 0
        self.cache_hits = 0
        self.tasks_completed_within_bound = 0

        # Task generation parameters
        self.num_iot_per_region = num_iot_per_region

        # ========== ENHANCED TASK GENERATION PARAMETERS (REPLACE OLD) ==========
        # Paper-compliant epoch-based parameters
        self.epoch_length = 5  # E slots per epoch (configurable)
        self.current_epoch = -1
        self.current_epoch_preferences = None  # π(e)_{x,y} from Dirichlet

        # Dirichlet concentration parameters (θ vector)
        self.theta_base = 1.0  # Base concentration
        self.theta_spatial_bias = 0.5  # Spatial preference bias
        self.theta_self_preference = 2.0  # Preference for own region

        # Spatiotemporal task Zipf parameters
        self.alpha_task_base = 1.5  # Base task Zipf parameter
        self.alpha_task_correlation = 0.3  # Correlation with IoT activation

        # Historical tracking for temporal patterns
        self.task_generation_history = {}  # Track temporal patterns

        # DO NOT ADD: self.spatial_distribution - THIS IS OLD AND REMOVED

        # Temporal state
        self.current_time = 0
        self.region = None  # Will be set by environment

        # Neighbor communication
        self.neighbor_links = {}

    def update_epoch_preferences(self, timestep):
        """
        Update region preferences using Dirichlet distribution per epoch
        Implements Paper Equation: π(e)_{x,y} ~ Dirichlet(θ)
        """
        current_epoch = timestep // self.epoch_length

        # Update preferences at the start of each epoch
        if current_epoch != self.current_epoch:
            self.current_epoch = current_epoch

            # Create concentration vector θ with spatial and temporal biases
            theta = np.ones(self.X * self.Y) * self.theta_base

            # Add spatial bias - preference for nearby regions
            x, y = self.pos
            for i in range(self.X):
                for j in range(self.Y):
                    region_idx = i * self.Y + j
                    # Distance-based bias
                    distance = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    theta[region_idx] += self.theta_spatial_bias * np.exp(-distance / 2.0)

            # Add self-preference (tasks more likely to target own region)
            own_region_idx = x * self.Y + y
            theta[own_region_idx] += self.theta_self_preference

            # Add temporal variation based on epoch
            temporal_factor = 0.5 * np.sin(2 * np.pi * current_epoch / 10)  # 10-epoch cycle
            theta += temporal_factor

            # Ensure all concentrations are positive
            theta = np.maximum(theta, 0.1)

            # Sample region preferences from Dirichlet distribution
            self.current_epoch_preferences = np.random.dirichlet(theta)

            print(f"UAV {self.pos}: Updated epoch {current_epoch} preferences - "
                  f"Own region preference: {self.current_epoch_preferences[own_region_idx]:.3f}")

    def get_spatiotemporal_task_zipf_parameter(self, target_region, timestep):
        """
        Calculate spatiotemporal Zipf parameter for task device selection
        Correlates with IoT activation patterns but with variation
        """
        target_x, target_y = target_region

        # Get IoT activation Zipf parameter if available
        if self.region and hasattr(self.region, 'current_zipf_param'):
            iot_alpha = self.region.current_zipf_param
        else:
            # Fallback calculation matching IoT region pattern
            # α_{x,y}(t) = a_base + φ_spatial(x,y) + φ_temporal(t,x,y)
            a_base = np.random.uniform(1.2, 2.5)
            phi_spatial = ((target_x * 3 + target_y * 2) % 5) * 0.3
            phi_temporal = 0.6 * np.sin((2 * np.pi * timestep / 20) + target_x + target_y)
            iot_alpha = a_base + phi_spatial + phi_temporal

        # Task Zipf parameter with correlation to IoT but slight variation
        alpha_task = (self.alpha_task_base +
                      self.alpha_task_correlation * iot_alpha +
                      0.2 * np.random.normal(0, 0.1))  # Small noise

        # Ensure reasonable bounds
        alpha_task = np.clip(alpha_task, 1.0, 3.0)

        return alpha_task

    def select_device_with_spatiotemporal_zipf(self, target_region, timestep):
        """
        Select IoT device using spatiotemporal Zipf distribution
        Implements proper device selection as described in paper
        """
        alpha_task = self.get_spatiotemporal_task_zipf_parameter(target_region, timestep)

        # Generate Zipf probabilities for device ranking
        ranks = np.arange(1, self.num_iot_per_region + 1)
        zipf_probs = np.power(ranks, -alpha_task)
        zipf_probs = zipf_probs / zipf_probs.sum()  # Normalize

        # Sample device according to Zipf distribution
        try:
            selected_device = np.random.choice(self.num_iot_per_region, p=zipf_probs)
            return selected_device
        except ValueError:
            # Fallback in case of numerical issues
            return np.random.randint(0, self.num_iot_per_region)


    # 3. COMPLETELY REPLACE THE generate_tasks METHOD:

    def generate_tasks(self, X, Y, timestep, num_tasks=5):
        """
        Generate content-aware tasks following paper model exactly
        Implements Paper Section D: Task Generation with epochs and Dirichlet distribution
        """
        # Update epoch preferences if needed
        self.update_epoch_preferences(timestep)

        tasks = []
        base_time = timestep * self.duration

        # Determine number of tasks to generate (with some randomness)
        num_tasks_to_generate = np.random.randint(max(1, num_tasks - 2), num_tasks + 3)

        # Generate tasks at random times within the slot
        if num_tasks_to_generate > self.duration:
            num_tasks_to_generate = self.duration

        task_offsets = sorted(np.random.choice(range(self.duration),
                                               size=num_tasks_to_generate,
                                               replace=False))

        # Spatial indices for all regions
        spatial_indices = [(i, j) for i in range(X) for j in range(Y)]

        for offset in task_offsets:
            # ========== PAPER-COMPLIANT REGION SELECTION ==========
            # Select target region using current epoch Dirichlet preferences
            region_idx = np.random.choice(len(spatial_indices),
                                          p=self.current_epoch_preferences)
            target_x, target_y = spatial_indices[region_idx]

            # ========== PAPER-COMPLIANT DEVICE SELECTION ==========
            # Select device using spatiotemporal Zipf distribution
            device_id = self.select_device_with_spatiotemporal_zipf(
                (target_x, target_y), timestep)

            # ========== PAPER-COMPLIANT TASK CREATION ==========
            # Create task following Equation 13: T(i)_{x,y}(t) = {ID(m)_{x',y'}, γ(i)_{x,y}, TTL(i)_{x,y}}
            task = {
                'task_id': np.random.randint(100000, 999999),
                'required_cpu': np.random.randint(1, 10),  # γ(i)_{x,y}(t) - CPU cycles
                'delay_bound': np.random.uniform(1.0, 15.0),  # TTL(i)_{x,y}(t) - deadline
                'content_id': (target_x, target_y, device_id),  # ID(m)_{x',y'} - content reference
                'generation_time': base_time + offset,  # Generation timestamp
                'size': np.random.uniform(0.5, 3.0),  # Task data size in MB
                'remaining_cpu': None,  # Will be set during processing

                # Additional metadata for analysis
                'epoch': self.current_epoch,
                'target_region_preference': self.current_epoch_preferences[region_idx],
                'alpha_task_used': self.get_spatiotemporal_task_zipf_parameter((target_x, target_y), timestep)
            }

            tasks.append(task)

        # Update generation history for analysis
        self.task_generation_history[timestep] = {
            'num_tasks': len(tasks),
            'epoch': self.current_epoch,
            'target_regions': [t['content_id'][:2] for t in tasks],
            'avg_preference': np.mean([t['target_region_preference'] for t in tasks]) if tasks else 0
        }

        print(f"UAV {self.pos}: Generated {len(tasks)} tasks for timestep {timestep} "
              f"(epoch {self.current_epoch})")

        # Print some analysis
        if tasks:
            own_region_tasks = sum(1 for t in tasks if t['content_id'][:2] == self.pos)
            cross_region_tasks = len(tasks) - own_region_tasks
            print(f"   Own region: {own_region_tasks}, Cross-region: {cross_region_tasks}")

        return tasks

    def get_task_generation_statistics(self):
        """
        Get statistics about task generation patterns for analysis
        """
        if not self.task_generation_history:
            return {}

        total_tasks = sum(h['num_tasks'] for h in self.task_generation_history.values())
        epochs_used = len(set(h['epoch'] for h in self.task_generation_history.values()))

        # Calculate region diversity
        all_targets = []
        for h in self.task_generation_history.values():
            all_targets.extend(h['target_regions'])

        unique_regions = len(set(all_targets))
        total_regions = self.X * self.Y

        return {
            'total_tasks_generated': total_tasks,
            'epochs_seen': epochs_used,
            'region_diversity': unique_regions / total_regions if total_regions > 0 else 0,
            'avg_tasks_per_timestep': total_tasks / len(
                self.task_generation_history) if self.task_generation_history else 0,
            'cross_region_ratio': sum(1 for target in all_targets if target != self.pos) / len(
                all_targets) if all_targets else 0
        }


    def aggregate_content(self, content_dict, interfering_regions):
        """
        Aggregate IoT content using proper TDMA protocol and energy model
        Implements paper communication model with interference
        """
        if not content_dict:
            self.aggregated_content = {}
            return

        aggregated = {}
        total_time_used = 0
        slot_duration = self.duration  # 300 seconds TDMA slot

        # Calculate interference from other regions
        interference = self.comm_model.estimate_co_channel_interference(
            source_regions=interfering_regions, target_region=self.pos)

        # Sort content by priority (smaller content first for efficiency)
        sorted_content = sorted(content_dict.items(), key=lambda x: x[1]['size'])

        for content_id, content in sorted_content:
            if total_time_used >= slot_duration:
                print(f"UAV {self.pos}: TDMA slot exhausted, remaining content dropped")
                break

            # Calculate transmission time with interference (Paper Equation 6)
            transmission_time = self.comm_model.calculate_transmission_delay(
                content_size=content['size'], interference=interference)

            if total_time_used + transmission_time <= slot_duration:
                # Content can be aggregated within slot
                aggregated[content_id] = content.copy()
                aggregated[content_id]['received_by_uav'] = total_time_used
                total_time_used += transmission_time

                # Calculate and deduct communication energy (Paper Equation 19)
                comm_energy = self.comm_model.calculate_communication_energy(
                    content_size=content['size'], transmission_time=transmission_time)
                self.energy -= comm_energy
                self.energy_used_this_slot += comm_energy

                print(f"UAV {self.pos}: Aggregated content {content_id} "
                      f"(size: {content['size']:.1f}MB, time: {transmission_time:.2f}s, "
                      f"energy: {comm_energy:.4f}J)")
            else:
                print(f"UAV {self.pos}: Content {content_id} too large for remaining TDMA slot")

        self.aggregated_content = aggregated
        print(f"UAV {self.pos}: Total aggregation time: {total_time_used:.2f}s/{slot_duration}s")


    def upload_to_satellite_with_proper_protocol(self, satellite, subchannel_assigned=False):
        """
        Upload content to satellite using Ka-band OFDMA (Paper Equations 8, 10, 11)
        """
        if not subchannel_assigned:
            print(f"UAV {self.pos}: No OFDM subchannel assigned for satellite upload")
            return []

        uploaded_content = []

        for cid, content in self.aggregated_content.items():
            # Use Ka-band uplink model (Equation 8)
            rate, success, delay_func = self.comm_model.compute_uav_to_satellite_uplink_rate(
                uav_pos=self.uav_pos,
                sat_pos=satellite.position,
                subchannel_assigned=True,
                fading=satellite.get_channel_state(self.pos)
            )

            if not success:
                print(f"UAV {self.pos}: Failed Ka-band uplink to satellite {satellite.sat_id}")
                continue

            # Calculate total delay (transmission + propagation) - Equations 10, 11
            total_upload_delay = delay_func(content['size'])

            # Create satellite content metadata
            sat_content = content.copy()
            sat_content['received_by_satellite'] = content['received_by_uav'] + total_upload_delay
            sat_content['upload_delay'] = total_upload_delay
            sat_content['uplink_rate'] = rate

            # Calculate uplink energy (Power × Time)
            content_bits = content['size'] * 8 * 1e6
            transmission_time = content_bits / rate
            uplink_power = self.comm_model.P_UAV  # 10 Watts
            uplink_energy = uplink_power * transmission_time

            # Deduct energy
            self.energy -= uplink_energy
            self.energy_used_this_slot += uplink_energy

            uploaded_content.append((cid, sat_content))

            print(f"UAV {self.pos}: Uploaded {cid} to satellite {satellite.sat_id}, "
                  f"delay={total_upload_delay:.3f}s, energy={uplink_energy:.2f}J")

        return uploaded_content

    def download_from_satellite(self, satellite, content_requests, subchannel_assigned=False):
        """
        Download content from satellite using Ka-band OFDMA (Paper Equation 9)
        """
        if not subchannel_assigned:
            return {}

        downloaded_content = {}

        for cid in content_requests:
            if cid not in satellite.global_content_set:
                continue

            # Use Ka-band downlink model (Equation 9)
            rate, success, delay_func = self.comm_model.compute_satellite_to_uav_downlink_rate(
                sat_pos=satellite.position,
                uav_pos=self.uav_pos,
                subchannel_assigned=True,
                fading=satellite.get_channel_state(self.pos)
            )

            if not success:
                continue

            # Assume content size from satellite metadata (simplified)
            content_size = 5.0  # MB placeholder
            total_download_delay = delay_func(content_size)

            # Create downloaded content
            downloaded_content[cid] = {
                'id': cid,
                'size': content_size,
                'downloaded_from_satellite': True,
                'download_delay': total_download_delay,
                'timestamp': self.current_time,
                'downlink_rate': rate
            }

            # Calculate reception energy (much lower than transmission)
            content_bits = content_size * 8 * 1e6
            transmission_time = content_bits / rate
            rx_power = 1.0  # 1 Watt for reception
            rx_energy = rx_power * transmission_time

            self.energy -= rx_energy
            self.energy_used_this_slot += rx_energy

        return downloaded_content

    def update_cache(self, timestamp, global_satellite_content_pool, is_connected_to_satellite):
        """
        Update cache using popularity-based selection (Paper Equation 22)
        Implements proper candidate pool construction
        """
        # Evict expired content first
        self.evict_expired_content(timestamp)

        # Build candidate pool C^cand_{x,y}(t) as per Equation 22
        candidate_pool = {}

        # Add existing cache content
        for cid, meta in self.cache_storage.items():
            candidate_pool[cid] = meta

        # Add aggregated content from this slot
        for cid, content in self.aggregated_content.items():
            candidate_pool[cid] = content

        # Add satellite content if connected (Equation 22 condition)
        if is_connected_to_satellite:
            for cid, sat_meta in global_satellite_content_pool.items():
                if cid not in candidate_pool:  # Don't override local content
                    meta = sat_meta.copy()
                    meta['timestamp'] = self.current_time
                    meta['from_satellite'] = True
                    candidate_pool[cid] = meta

        # Sort by popularity (usefulness scores)
        popular_items = []
        regular_items = []

        for cid, meta in candidate_pool.items():
            if cid in self.content_popularity:
                popular_items.append((cid, self.content_popularity[cid], meta))
            else:
                regular_items.append((cid, meta))

        # Sort popular items by popularity (descending)
        popular_items.sort(key=lambda x: x[1], reverse=True)

        # Build new cache within capacity constraint (Equation 29)
        new_cache = {}
        used_mb = 0.0

        # First, add popular items
        for cid, popularity, meta in popular_items:
            size = meta['size']
            if used_mb + size <= self.cache_capacity_mb:
                new_cache[cid] = meta
                used_mb += size

        # Then add remaining items if space available
        for cid, meta in regular_items:
            if cid in new_cache:
                continue
            size = meta['size']
            if used_mb + size <= self.cache_capacity_mb:
                new_cache[cid] = meta
                used_mb += size

        # Update cache
        self.cache_storage = new_cache
        self.cache_used_mb = used_mb

        print(f"UAV {self.pos}: Cache updated - {len(new_cache)} items, {used_mb:.1f}/{self.cache_capacity_mb}MB")

    def evict_expired_content(self, timestamp):
        """
        Remove content that has exceeded its TTL
        """
        current_time = (timestamp + 1) * self.duration
        expired = []

        for cid, meta in self.cache_storage.items():
            if current_time - meta['generation_time'] > meta['ttl']:
                expired.append(cid)

        for cid in expired:
            self.cache_used_mb -= self.cache_storage[cid]['size']
            del self.cache_storage[cid]
            if cid in self.content_popularity:
                # Decay popularity of expired content
                self.content_popularity[cid] *= 0.5

        if expired:
            print(f"UAV {self.pos}: Evicted {len(expired)} expired content items")

    def receive_task(self, task, from_coord):
        """
        Receive task and check content availability
        """
        cid = tuple(task['content_id'])

        # Check if required content is available locally
        if cid not in self.cache_storage and cid not in self.aggregated_content:
            print(f"UAV {self.pos}: Dropped task {task['task_id']} - content {cid} not available")
            return False

        # Update content popularity
        self.content_popularity[cid] = self.content_popularity.get(cid, 0) + 1

        # Add receive time jitter (1-10ms)
        receive_jitter = np.random.uniform(0.001, 0.010)
        task['receive_time'] = task['generation_time'] + receive_jitter
        task['from_coord'] = from_coord

        # Add to queue
        self.queue.append(task)
        self.queue.sort(key=lambda t: t['receive_time'])  # Sort by receive time

        print(f"UAV {self.pos}: Queued task {task['task_id']} at {task['receive_time']:.3f}s")
        return True

    def execute_tasks(self, timestep):
        """
        Execute tasks with proper computation energy model (Paper Equation 20)
        """
        completed = []
        current_time = getattr(self, 'next_available_time', timestep * self.duration)

        for task in list(self.queue):
            cid = tuple(task['content_id'])
            content_ready_time = None

            # Check content availability and readiness
            if timestep > 0 and cid in self.cache_storage:
                content_ready_time = timestep * self.duration
            elif cid in self.aggregated_content:
                content = self.aggregated_content[cid]
                content_ready_time = content.get('received_by_uav', 0)

            if content_ready_time is None:
                print(f"UAV {self.pos}: Skipped task {task['task_id']} - content not ready")
                self.queue.remove(task)
                continue

            # Calculate execution timing
            start_time = max(task['receive_time'], current_time)
            compute_time = task['required_cpu'] / self.compute_power  # Time = Cycles / Frequency
            completion_time = start_time + compute_time
            total_delay = completion_time - task['generation_time']

            # Check delay bound constraint (Equation 28)
            if total_delay > task['delay_bound']:
                print(f"UAV {self.pos}: Dropped task {task['task_id']} - "
                      f"delay bound exceeded ({total_delay:.2f}s > {task['delay_bound']:.2f}s)")
                self.queue.remove(task)
                continue

            # Calculate computation energy (Paper Equation 20)
            # E^cp_{x,y}(t) = ξ × Σγ^(j)_{x,y}(t) × (f^comp_{x,y})²
            xi = 1e-10  # Effective switched capacitance
            cpu_cycles = task['required_cpu']
            frequency = self.compute_power
            computation_energy = xi * cpu_cycles * (frequency ** 2)

            # Deduct energy
            self.energy -= computation_energy
            self.energy_used_this_slot += computation_energy

            # Update task with execution details
            task['delay'] = {
                'queue_delay': start_time - task['receive_time'],
                'computation_delay': compute_time,
                'total_delay': total_delay,
                'execution_start': start_time,
                'completion_time': completion_time
            }
            task['energy_consumed'] = computation_energy

            # Update UAV state
            current_time = completion_time
            self.next_available_time = current_time
            self.cache_hits += int(cid in self.cache_storage)
            completed.append(task)
            self.queue.remove(task)
            self.tasks_completed_within_bound += 1

            print(f"UAV {self.pos}: Executed task {task['task_id']} - "
                  f"completed at {completion_time:.2f}s, energy={computation_energy:.4f}J")

        return completed


    def clear_aggregated_content(self):
        """Clear aggregated IoT content after caching decision"""
        self.aggregated_content = {}

    def observe(self, neighbor_loads, satellite_in_range, activation_mask):
        """
        Create observation vector for RL agents
        """
        obs = []
        # Energy state (normalized)
        obs.append(self.energy / self.max_energy)
        # Queue state (normalized)
        obs.append(len(self.queue) / self.max_queue)
        # Cache utilization
        obs.append(self.cache_used_mb / self.cache_capacity_mb)

        # Cache content vector (simplified)
        cache_vec = np.zeros(len(activation_mask))
        for i, active in enumerate(activation_mask):
            if i < len(cache_vec):
                cache_vec[i] = active
        obs.extend(cache_vec)

        # Neighbor information
        obs.extend(neighbor_loads.flatten())
        # Satellite connectivity
        obs.extend(satellite_in_range)
        # Current activation pattern
        obs.extend(activation_mask)

        return np.array(obs, dtype=np.float32)

    def get_energy_efficiency(self):
        """Calculate energy efficiency metric"""
        if self.energy_used_this_slot == 0:
            return float('inf')
        return self.tasks_completed_within_bound / self.energy_used_this_slot

    def reset_slot_counters(self):
        """Reset per-slot counters"""
        self.energy_used_this_slot = 0.0

    def get_status_summary(self):
        """Get current UAV status for monitoring"""
        return {
            'position': self.pos,
            'energy': self.energy,
            'energy_used_this_slot': self.energy_used_this_slot,
            'cache_usage': f"{self.cache_used_mb:.1f}/{self.cache_capacity_mb}MB",
            'queue_length': len(self.queue),
            'tasks_completed': self.tasks_completed_within_bound,
            'cache_hits': self.cache_hits,
            'total_tasks': self.total_tasks
        }


# Testing
if __name__ == "__main__":
    print("=== Testing UAV Class ===")

    # Create test UAV
    uav = UAV(x=1, y=1, X=3, Y=3, cache_size=50, max_queue=10,
              duration=300, compute_power=20, energy=100000, num_iot_per_region=15)

    print(f"UAV Status: {uav.get_status_summary()}")

    # Test task generation
    tasks = uav.generate_tasks(3, 3, timestep=0, num_tasks=3)
    print(f"Generated {len(tasks)} tasks")

    # Test energy efficiency
    efficiency = uav.get_energy_efficiency()
    print(f"Energy efficiency: {efficiency}")

    print("\n=== UAV Test Complete ===")