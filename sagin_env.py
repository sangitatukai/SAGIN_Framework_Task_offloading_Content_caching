# sagin_env.py - COMPLETE REPLACEMENT
import numpy as np
from iot_region import IoTRegion
from satellite import Satellite
from uav import UAV
from communication_model import CommunicationModel


class SystemDownException(Exception):
    """Exception raised when UAV energy is depleted"""
    pass


class SAGINEnv:
    def __init__(self, X, Y, duration, cache_size, compute_power_uav, compute_power_sat,
                 energy, max_queue, num_sats, num_iot_per_region, max_active_iot, ofdm_slots):
        # Grid dimensions and timing
        self.X, self.Y = X, Y
        self.duration = duration  # Time slot duration (seconds)

        # Communication model
        self.comm_model = CommunicationModel()

        # Create UAVs for each grid position
        self.uavs = {}
        for x in range(X):
            for y in range(Y):
                uav = UAV(x, y, X, Y, cache_size, max_queue, duration,
                          compute_power_uav, energy, num_iot_per_region)
                self.uavs[(x, y)] = uav

        # Create IoT regions for each grid position
        self.iot_regions = {}
        for x in range(X):
            for y in range(Y):
                region = IoTRegion(num_iot_per_region, max_active_iot, duration,
                                   region_coords=(x, y))
                self.iot_regions[(x, y)] = region
                # Link UAV to its region
                self.uavs[(x, y)].region = region

        # Create satellites
        self.sats = []
        for s in range(num_sats):
            # LEO satellite positions (simplified circular orbit)
            position = (s * 200, 500, 550000)  # 550km altitude
            coverage = self._generate_universal_coverage()
            satellite = Satellite(s, storage_capacity=1000, coverage_map=coverage,
                                  duration=duration, position=position,
                                  compute_power=compute_power_sat)
            self.sats.append(satellite)

        # OFDM slot management (Paper Equation 23)
        self.ofdm_slots = ofdm_slots  # Total available OFDM subchannels
        self.subchannel_assignments = {}  # {uav_coord: {sat_id: bool}}
        self.connected_uavs = set()  # UAVs with assigned subchannels

        # Global satellite content pool
        self.global_satellite_content_pool = {}  # Global pool G(t)

        # Logging and statistics
        self.task_log = []
        self.success_log = []
        self.dropped_tasks = 0
        self.g_timestep = -1  # Global timestep counter

        # Performance tracking
        self.task_stats = {
            'uav': {},  # {(x,y): {'generated': int, 'completed': int, 'successful': int}}
            'satellite': {}  # {sat_id: {'received': int, 'completed': int, 'successful': int}}
        }

        print(f"SAGIN Environment initialized: {X}x{Y} grid, {num_sats} satellites, {ofdm_slots} OFDM slots")

    def _generate_universal_coverage(self):
        """Generate universal coverage map for satellites"""
        return [{(i, j) for i in range(self.X) for j in range(self.Y)} for _ in range(10)]

    def collect_iot_data(self):
        """
        Phase 1: IoT data collection using TDMA protocol with proper interference
        Implements paper IoT activation and content aggregation model
        """
        self.g_timestep += 1
        print(f"\n=== Timestep {self.g_timestep}: IoT Data Collection ===")

        for (x, y), uav in self.uavs.items():
            # Get active IoT devices using spatiotemporal Zipf distribution (Equations 1-3)
            active_device_ids = self.iot_regions[(x, y)].sample_active_devices()

            if not active_device_ids:
                print(f"UAV ({x},{y}): No active IoT devices")
                uav.aggregated_content = {}
                continue

            # Generate content from active devices (Equation 4)
            content_list = self.iot_regions[(x, y)].generate_content(
                active_device_ids, self.g_timestep, grid_coord=(x, y)
            )
            content_dict = {tuple(c['id']): c for c in content_list}

            # Calculate interfering regions for co-channel interference
            interfering_regions = [coord for coord in self.uavs.keys() if coord != (x, y)]

            # Aggregate content with proper TDMA protocol and energy model
            uav.aggregate_content(content_dict, interfering_regions)

            print(f"UAV ({x},{y}): Active devices: {len(active_device_ids)}, "
                  f"Aggregated content: {len(uav.aggregated_content)}")

    def allocate_ofdm_slots_with_constraints(self):
        """
        Phase 2: OFDM slot allocation with proper constraint enforcement
        Implements Paper Equation (23): Σ Σ Σ θ_{x,y,k}(t) ≤ S_OFDMA
        """
        print(f"\n=== Timestep {self.g_timestep}: OFDM Slot Allocation ===")

        # Reset subchannel assignments
        self.subchannel_assignments = {coord: {} for coord in self.uavs.keys()}
        self.connected_uavs = set()

        # Priority-based allocation (can be replaced with RL later)
        # Calculate priority based on aggregated content size and queue length
        uav_priorities = []
        for coord, uav in self.uavs.items():
            aggregated_size = sum(content['size'] for content in uav.aggregated_content.values())
            queue_urgency = len(uav.queue)
            priority = aggregated_size + queue_urgency * 2  # Weight queue more heavily
            uav_priorities.append((priority, coord, uav))

        # Sort by priority (descending)
        uav_priorities.sort(key=lambda x: x[0], reverse=True)

        # Allocate subchannels ensuring constraint compliance
        allocated_channels = 0
        sat_assignments = {sat.sat_id: [] for sat in self.sats}

        for priority, coord, uav in uav_priorities:
            if allocated_channels >= self.ofdm_slots:
                print(f"OFDM slots exhausted ({allocated_channels}/{self.ofdm_slots})")
                break

            # Find satellite with available capacity
            best_sat = None
            min_load = float('inf')

            for sat in self.sats:
                current_load = len(sat_assignments[sat.sat_id])
                if current_load < min_load:
                    min_load = current_load
                    best_sat = sat

            if best_sat is not None:
                # Assign subchannel
                self.subchannel_assignments[coord][best_sat.sat_id] = True
                sat_assignments[best_sat.sat_id].append(coord)
                self.connected_uavs.add(coord)
                allocated_channels += 1

                print(f"Assigned UAV {coord} to Satellite {best_sat.sat_id} "
                      f"(priority={priority:.1f}, subchannel {allocated_channels}/{self.ofdm_slots})")

        # Verify constraint compliance (Paper Equation 23)
        total_assignments = sum(
            sum(assignments.values()) for assignments in self.subchannel_assignments.values()
        )

        if total_assignments > self.ofdm_slots:
            raise ValueError(f"OFDM constraint violated: {total_assignments} > {self.ofdm_slots}")

        print(f"OFDM Allocation complete: {total_assignments}/{self.ofdm_slots} subchannels used")

    def upload_to_satellites(self):
        """
        Phase 3: UAV-to-Satellite upload using Ka-band OFDMA
        Implements proper satellite communication protocols
        """
        print(f"\n=== Timestep {self.g_timestep}: Satellite Upload ===")

        for (x, y), uav in self.uavs.items():
            # Check if UAV has subchannel assignment
            uav_assignments = self.subchannel_assignments.get((x, y), {})
            assigned_satellite = None

            for sat in self.sats:
                if uav_assignments.get(sat.sat_id, False):
                    assigned_satellite = sat
                    break

            if assigned_satellite is None:
                print(f"UAV ({x},{y}): No subchannel assigned, skipping upload")
                continue

            # Upload using proper Ka-band protocol
            uploaded_content = uav.upload_to_satellite_with_proper_protocol(
                satellite=assigned_satellite,
                subchannel_assigned=True
            )

            # Update global satellite content pool G(t)
            for cid, sat_content in uploaded_content:
                self.global_satellite_content_pool[cid] = sat_content
                print(f"Added {cid} to global satellite pool")

            if uploaded_content:
                print(f"UAV ({x},{y}): Uploaded {len(uploaded_content)} items to Satellite {assigned_satellite.sat_id}")

    def sync_satellites(self):
        """
        Phase 4: Synchronize satellite content pools
        Implements global content pool G(t) synchronization
        """
        print(f"\n=== Timestep {self.g_timestep}: Satellite Synchronization ===")

        # Update all satellites with global content pool
        global_content_ids = set(self.global_satellite_content_pool.keys())

        for sat in self.sats:
            sat.global_content_set = global_content_ids.copy()
            print(f"Satellite {sat.sat_id}: Synchronized with {len(global_content_ids)} global content items")

    def generate_and_offload_tasks(self):
        """
        Phase 5: Task generation and offloading decisions
        Implements content-aware task offloading
        """
        print(f"\n=== Timestep {self.g_timestep}: Task Generation & Offloading ===")

        for (x, y), uav in self.uavs.items():
            # Generate content-aware tasks
            tasks = uav.generate_tasks(self.X, self.Y, self.g_timestep)

            # Initialize task stats if needed
            if (x, y) not in self.task_stats['uav']:
                self.task_stats['uav'][(x, y)] = {'generated': 0, 'completed': 0, 'successful': 0}

            # Process each generated task
            for task in tasks:
                task['remaining_cpu'] = task['required_cpu']
                self.task_stats['uav'][(x, y)]['generated'] += 1
                uav.total_tasks += 1

                # Make offloading decision
                offload_target = self.make_offloading_decision(task, (x, y), uav)

                # Log task assignment
                self.task_log.append({
                    "timestep": self.g_timestep,
                    "task_id": task['task_id'],
                    "generated_by": (x, y),
                    "assigned_to": offload_target,
                    "content_id": task['content_id'],
                    "delay_bound": task['delay_bound']
                })

    def make_offloading_decision(self, task, uav_coord, uav):
        """
        Make intelligent offloading decision based on content availability
        Baseline approach: content-aware offloading with energy consideration
        """
        x, y = uav_coord
        cid = tuple(task['content_id'])
        task_id = task['task_id']

        # Option 1: Local execution (UAV has content)
        if cid in uav.cache_storage or cid in uav.aggregated_content:
            success = uav.receive_task(task, from_coord=(x, y))
            if success:
                print(f"[TASK {task_id}] Content {cid} found locally at UAV ({x},{y}) — assigned LOCAL")
                return 'local'

        # Option 2: Neighbor UAV execution (check 4-connected neighbors)
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_neighbor = None
        min_queue_length = float('inf')

        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.uavs:
                continue

            neighbor = self.uavs[(nx, ny)]

            # Check if neighbor has content and capacity
            if (cid in neighbor.cache_storage or cid in neighbor.aggregated_content) and \
                    len(neighbor.queue) < neighbor.max_queue and \
                    neighbor.energy > neighbor.max_energy * 0.2:  # At least 20% energy

                if len(neighbor.queue) < min_queue_length:
                    min_queue_length = len(neighbor.queue)
                    best_neighbor = neighbor
                    best_neighbor_coord = (nx, ny)

        if best_neighbor is not None:
            success = best_neighbor.receive_task(task, from_coord=(x, y))
            if success:
                print(f"[TASK {task_id}] Content {cid} found at NEIGHBOR UAV {best_neighbor_coord} — offloaded")
                return f'neighbor_{best_neighbor_coord[0]}_{best_neighbor_coord[1]}'

        # Option 3: Satellite execution (if UAV is connected and satellite has content)
        if (x, y) in self.connected_uavs:
            for sat in self.sats:
                sat_assignments = self.subchannel_assignments.get((x, y), {})
                if sat_assignments.get(sat.sat_id, False) and cid in self.global_satellite_content_pool:
                    sat.receive_task(task, from_coord=uav.uav_pos)

                    # Update satellite stats
                    if sat.sat_id not in self.task_stats['satellite']:
                        self.task_stats['satellite'][sat.sat_id] = {'received': 0, 'completed': 0, 'successful': 0}
                    self.task_stats['satellite'][sat.sat_id]['received'] += 1

                    print(f"[TASK {task_id}] Content {cid} found in SATELLITE {sat.sat_id} — offloaded")
                    return f'satellite_{sat.sat_id}'

        # Option 4: Drop task (content not available anywhere)
        self.dropped_tasks += 1
        print(f"[TASK {task_id}] Content {cid} NOT found anywhere — DROPPED")
        return 'dropped'

    def step(self):
        """
        Execute one simulation step with all phases
        """
        print(f"\n{'=' * 60}")
        print(f"SAGIN STEP {self.g_timestep + 1}")
        print(f"{'=' * 60}")

        # Phase 1: IoT Data Collection
        self.collect_iot_data()

        # Phase 2: OFDM Slot Allocation
        self.allocate_ofdm_slots_with_constraints()

        # Phase 3: Update satellite coverage
        for sat in self.sats:
            sat.update_coverage(self.g_timestep)

        # Phase 4: UAV-to-Satellite Upload
        self.upload_to_satellites()

        # Phase 5: Satellite Synchronization
        self.sync_satellites()

        # Phase 6: Task Generation and Offloading
        self.generate_and_offload_tasks()

        # Phase 7: Task Execution
        self.execute_all_tasks()

        # Phase 8: Cache Updates and Maintenance
        self.update_all_caches()

        # Phase 9: Content Eviction
        self.evict_expired_content()

        # Phase 10: Energy and System Monitoring
        self.monitor_system_health()

    def execute_all_tasks(self):
        """Execute tasks on all UAVs and satellites"""
        print(f"\n=== Timestep {self.g_timestep}: Task Execution ===")

        # Execute UAV tasks
        for (x, y), uav in self.uavs.items():
            completed = uav.execute_tasks(self.g_timestep)

            # Count successful completions
            successful = sum(
                1 for t in completed
                if t.get('delay', {}).get('total_delay', float('inf')) <= t.get('delay_bound', float('inf'))
            )

            # Update statistics
            if (x, y) not in self.task_stats['uav']:
                self.task_stats['uav'][(x, y)] = {'generated': 0, 'completed': 0, 'successful': 0}

            self.task_stats['uav'][(x, y)]['completed'] += len(completed)
            self.task_stats['uav'][(x, y)]['successful'] += successful

            if completed:
                print(f"UAV ({x},{y}): Completed {len(completed)} tasks, {successful} within deadline")

        # Execute satellite tasks
        for sat in self.sats:
            completed = sat.execute_tasks(self.g_timestep, self.global_satellite_content_pool)

            # Count successful completions
            successful = sum(
                1 for t in completed
                if t.get('delay', {}).get('total_delay', float('inf')) <= t.get('delay_bound', float('inf'))
            )

            # Update statistics
            if sat.sat_id not in self.task_stats['satellite']:
                self.task_stats['satellite'][sat.sat_id] = {'received': 0, 'completed': 0, 'successful': 0}

            self.task_stats['satellite'][sat.sat_id]['completed'] += len(completed)
            self.task_stats['satellite'][sat.sat_id]['successful'] += successful

            if completed:
                print(f"Satellite {sat.sat_id}: Completed {len(completed)} tasks, {successful} within deadline")

    def update_all_caches(self):
        """Update caches on all UAVs"""
        print(f"\n=== Timestep {self.g_timestep}: Cache Updates ===")

        for (x, y), uav in self.uavs.items():
            is_connected = (x, y) in self.connected_uavs
            uav.update_cache(
                timestamp=self.g_timestep,
                global_satellite_content_pool=self.global_satellite_content_pool,
                is_connected_to_satellite=is_connected
            )
            uav.clear_aggregated_content()

    def evict_expired_content(self):
        """Evict expired content from global satellite pool"""
        current_time = (self.g_timestep + 1) * self.duration
        expired = []

        for cid, meta in list(self.global_satellite_content_pool.items()):
            if current_time - meta.get('generation_time', 0) > meta.get('ttl', float('inf')):
                expired.append(cid)

        for cid in expired:
            del self.global_satellite_content_pool[cid]

        if expired:
            print(f"Evicted {len(expired)} expired items from global satellite pool")

    def monitor_system_health(self):
        """Monitor system health and energy levels"""
        print(f"\n=== Timestep {self.g_timestep}: System Health Check ===")

        for (x, y), uav in self.uavs.items():
            # Check energy levels
            energy_percentage = (uav.energy / uav.max_energy) * 100

            if uav.energy <= 0:
                print(f"[CRITICAL] UAV at ({x}, {y}) has depleted energy!")
                raise SystemDownException(f"UAV at ({x}, {y}) energy depleted. SYSTEM DOWN.")
            elif energy_percentage < 10:
                print(f"[WARNING] UAV at ({x}, {y}) low energy: {energy_percentage:.1f}%")

            # Reset per-slot counters
            uav.reset_slot_counters()

            # Print status summary
            status = uav.get_status_summary()
            print(f"UAV ({x},{y}): Energy={energy_percentage:.1f}%, "
                  f"Cache={status['cache_usage']}, Queue={status['queue_length']}, "
                  f"Tasks={status['tasks_completed']}")

    def get_performance_summary(self):
        """Get overall system performance summary"""
        # Calculate success rates
        total_uav_generated = sum(stats['generated'] for stats in self.task_stats['uav'].values())
        total_uav_successful = sum(stats['successful'] for stats in self.task_stats['uav'].values())
        uav_success_rate = (total_uav_successful / total_uav_generated * 100) if total_uav_generated > 0 else 0

        total_sat_received = sum(stats['received'] for stats in self.task_stats['satellite'].values())
        total_sat_successful = sum(stats['successful'] for stats in self.task_stats['satellite'].values())
        sat_success_rate = (total_sat_successful / total_sat_received * 100) if total_sat_received > 0 else 0

        # Calculate energy efficiency
        total_energy_used = sum(uav.max_energy - uav.energy for uav in self.uavs.values())
        total_tasks_completed = total_uav_successful + total_sat_successful
        energy_efficiency = total_tasks_completed / total_energy_used if total_energy_used > 0 else 0

        # Calculate cache hit rates
        cache_hits = sum(uav.cache_hits for uav in self.uavs.values())
        total_tasks = sum(uav.total_tasks for uav in self.uavs.values())
        cache_hit_rate = (cache_hits / total_tasks * 100) if total_tasks > 0 else 0

        return {
            'timestep': self.g_timestep,
            'uav_success_rate': uav_success_rate,
            'satellite_success_rate': sat_success_rate,
            'overall_success_rate': ((total_uav_successful + total_sat_successful) /
                                     (total_uav_generated + total_sat_received) * 100) if (
                                                                                                      total_uav_generated + total_sat_received) > 0 else 0,
            'cache_hit_rate': cache_hit_rate,
            'energy_efficiency': energy_efficiency,
            'dropped_tasks': self.dropped_tasks,
            'total_energy_used': total_energy_used,
            'global_content_pool_size': len(self.global_satellite_content_pool)
        }

    def print_final_summary(self):
        """Print comprehensive final summary"""
        print(f"\n{'=' * 80}")
        print(f"FINAL SAGIN SIMULATION SUMMARY (Timestep {self.g_timestep})")
        print(f"{'=' * 80}")

        summary = self.get_performance_summary()

        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Success Rate: {summary['overall_success_rate']:.2f}%")
        print(f"   Cache Hit Rate: {summary['cache_hit_rate']:.2f}%")
        print(f"   Energy Efficiency: {summary['energy_efficiency']:.4f} tasks/Joule")
        print(f"   Dropped Tasks: {summary['dropped_tasks']}")

        print(f"\n🛰️ UAV PERFORMANCE:")
        for coord, stats in self.task_stats['uav'].items():
            if stats['generated'] > 0:
                rate = stats['successful'] / stats['generated'] * 100
                print(f"   UAV {coord}: {stats['successful']}/{stats['generated']} successful ({rate:.1f}%)")

        print(f"\n🛸 SATELLITE PERFORMANCE:")
        for sat_id, stats in self.task_stats['satellite'].items():
            if stats['received'] > 0:
                rate = stats['successful'] / stats['received'] * 100
                print(f"   Satellite {sat_id}: {stats['successful']}/{stats['received']} successful ({rate:.1f}%)")

        print(f"\n⚡ ENERGY STATUS:")
        for coord, uav in self.uavs.items():
            energy_pct = (uav.energy / uav.max_energy) * 100
            print(f"   UAV {coord}: {energy_pct:.1f}% energy remaining")


# Testing
if __name__ == "__main__":
    print("=== Testing SAGIN Environment ===")

    # Create test environment
    env = SAGINEnv(X=3, Y=3, duration=300, cache_size=50, compute_power_uav=20,
                   compute_power_sat=200, energy=50000, max_queue=10, num_sats=2,
                   num_iot_per_region=15, max_active_iot=8, ofdm_slots=6)

    # Run a few timesteps
    try:
        for i in range(3):
            env.step()
            summary = env.get_performance_summary()
            print(f"\nTimestep {i}: Success Rate = {summary['overall_success_rate']:.2f}%")
    except SystemDownException as e:
        print(f"System shutdown: {e}")

    # Print final summary
    env.print_final_summary()

    print("\n=== SAGIN Environment Test Complete ===")