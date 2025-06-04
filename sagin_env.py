
import numpy as np
from iot_region import IoTRegion
from satellite import Satellite
from uav import UAV
from communication_model import compute_rate_general

class SystemDownException(Exception):
    pass

class SAGINEnv:
    def __init__(self, X, Y, duration, cache_size, compute_power_uav, compute_power_sat, energy, max_queue, num_sats, num_iot_per_region, max_active_iot, ofdm_slots):
        self.X, self.Y = X, Y
        self.uavs = {(x, y): UAV(x, y, X, Y, cache_size, max_queue, duration, compute_power_uav, energy, num_iot_per_region) for x in range(X) for y in range(Y)}
        self.iot_regions = {(x, y): IoTRegion(num_iot_per_region, max_active_iot, duration, region_coords=(x, y)) for x in range(X) for y in range(Y)}
        for coord in self.uavs:
            self.uavs[coord].region = self.iot_regions[coord]
        self.sats = [Satellite(s, 1000, self._generate_coverage(), duration, position=(s*100, 500, 550000), compute_power=compute_power_sat) for s in range(num_sats)]
        self.duration = duration
        self.ofdm_slots = ofdm_slots
        self.connected_uavs = set()
        self.global_satellite_content_pool = {}  # cid → metadata dict
        self.task_log = []
        self.dropped_tasks = 0
        self.success_log = []
        self.g_timestep = -1
        self.current_time=0
        self.task_stats = {
            'uav': {},  # {(x,y): {'generated': int, 'completed': int, 'successful': int}}
            'satellite': {}  # {sat_id: {'received': int, 'completed': int, 'successful': int}}
        }

    def collect_iot_data(self):
        self.g_timestep += 1
        for (x, y), uav in self.uavs.items():
            predicted = self.iot_regions[(x, y)].sample_active_devices()
            content_list = self.iot_regions[(x, y)].generate_content(predicted, self.g_timestep, grid_coord=(x, y))
            content_dict = {tuple(c['id']): c for c in content_list}
            uav.aggregate_content(content_dict)

    def generate_and_offload_tasks(self):
        for (x, y), uav in self.uavs.items(): #this loop generate and offload all the task
            tasks = uav.generate_tasks(self.X, self.Y, self.g_timestep)
            for task in tasks:
                task['remaining_cpu'] = task['required_cpu']
                offload_target = self.offload_task(task, (x, y), uav.uav_pos)
                uav.total_tasks += 1
                self.task_log.append({"task_id": task['task_id'], "assigned_to": offload_target})

                if (x, y) not in self.task_stats['uav']:
                    self.task_stats['uav'][(x, y)] = {'generated': 0, 'completed': 0, 'successful': 0}
                self.task_stats['uav'][(x, y)]['generated'] += 1

    def _generate_coverage(self):
        return [{(i, j) for i in range(self.X) for j in range(self.Y)} for _ in range(10)]



    def allocate_ofdm_slots(self):
        self.satellite_slots = {sat.sat_id: [] for sat in self.sats}
        keys = list(self.uavs.keys())
        np.random.shuffle(keys)
        self.connected_uavs = set()
        slot_limit = self.ofdm_slots // len(self.sats)
        i = 0
        for sat in self.sats:
            self.satellite_slots[sat.sat_id] = keys[i:i+slot_limit]
            self.connected_uavs.update(keys[i:i+slot_limit])
            i += slot_limit

    def sync_satellites(self):
        for sat in self.sats:
            sat.global_content_set = set(self.global_satellite_content_pool.keys())

    def upload_to_satellites(self):
        for (x, y), uav in self.uavs.items():
            if (x, y) not in self.connected_uavs:
                continue
            uav_pos = (x * 100 + 50, y * 100 + 50, 100)
            for sat in self.sats:
                if (x, y) not in self.satellite_slots[sat.sat_id]:
                    continue
                sat_pos = sat.position
                d = np.linalg.norm(np.array(uav_pos) - np.array(sat_pos))
                propagation_speed = 3e8
                propagation_delay = d / propagation_speed

                for content in uav.aggregated_content.values():
                    # Transmission delay based on data rate
                    content_bits = content['size'] * 8 * 1e6
                    fading = sat.get_channel_state((x, y))
                    rate, ok = compute_rate_general(
                        sender_pos=uav_pos,
                        receiver_pos=sat_pos,
                        bandwidth=1e6,  # 1 MHz per slot
                        P_tx=10.0,  # UAV high-end power
                        fc=12e9,  # Ku-band
                        G_tx=10000,  # 40 dB UAV directional
                        G_rx=10000,  # 40 dB Sat antenna
                        noise=1e-13,  # Ultra low noise
                        fading=1.0
                    )
                    if not ok:
                        print("hi--sat")
                        continue

                    transmission_delay = content_bits / rate

                    # Energy usage: 2 units per second of transmission
                    energy_used = int(np.ceil(transmission_delay)) * 2
                    uav.energy -= energy_used
                    uav.energy_used_this_slot += energy_used

                    # Prepare satellite content metadata
                    cid = tuple(content['id'])
                    sat_content = content.copy()  # Copy all fields
                    sat_content['received_by_satellite'] = content['received_by_uav'] + transmission_delay + propagation_delay
                    print(f"trans delay:{transmission_delay}, propagration: {propagation_delay}, receive: {sat_content['received_by_satellite']}")

                    self.global_satellite_content_pool[cid] = sat_content

                break  # Only one satellite per UAV

    def evict_expired_content(self):
        expired = []
        seen = set()
        duplicates = []

        current_time = (self.g_timestep + 1) * self.duration

        # Find expired and duplicate content
        for cid, meta in list(self.global_satellite_content_pool.items()):
            # Expired if past TTL
            if current_time - meta.get('generation_time', 0) > meta.get('ttl', float('inf')):
                expired.append(cid)

        # Delete expired content
        for cid in expired:
            del self.global_satellite_content_pool[cid]


        print(f"Evicted {len(expired)} expired content items from global_satellite_content_pool.")

    def offload_task(self, task, uav_coord, uav_pos):
        x, y = uav_coord
        uav = self.uavs[(x, y)]
        cid = tuple(task['content_id'])
        task_id = task['task_id']

        if cid in uav.cache_storage or cid in uav.aggregated_content:
            uav.receive_task(task, from_coord=(x, y))
            print(f"[TASK {task_id}] Content {cid} found locally at UAV ({x},{y}) — assigned LOCAL")
            return 'local'

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.uavs:
                neighbor = self.uavs[(nx, ny)]
                if cid in neighbor .cache_storage or cid in neighbor.aggregated_content:
                    neighbor.receive_task(task, from_coord=(x, y))
                    print(f"[TASK {task_id}] Content {cid} found at NEIGHBOR UAV ({nx},{ny}) — offloaded")
                    return f'neighbor_{nx}_{ny}'

        if (x, y) in self.connected_uavs:
            for sat in self.sats:
                if (x, y) in self.satellite_slots.get(sat.sat_id, []) and cid in self.global_satellite_content_pool:
                    sat.receive_task(task, from_coord=uav_pos)
                    if sat.sat_id not in self.task_stats['satellite']:
                        self.task_stats['satellite'][sat.sat_id] = {'received': 0, 'completed': 0, 'successful': 0}
                    self.task_stats['satellite'][sat.sat_id]['received'] += 1
                    print(f"[TASK {task_id}] Content {cid} found in SATELLITE {sat.sat_id} — offloaded")
                    return f'satellite_{sat.sat_id}'


        self.dropped_tasks += 1
        print(f"[TASK {task_id}] Content {cid} NOT found anywhere — DROPPED")
        return 'dropped'

    def step(self):
        for (x, y), uav in self.uavs.items():
            # === UAV executes tasks ===
            completed = uav.execute_tasks(self.g_timestep)
            success = sum(
                t.get('delay', {}).get('total_delay', float('inf')) <= t.get('delay_bound', float('inf'))
                for t in completed
            )

            # === Initialize UAV stats if not present ===
            if (x, y) not in self.task_stats['uav']:
                self.task_stats['uav'][(x, y)] = {'generated': 0, 'completed': 0, 'successful': 0}

            self.task_stats['uav'][(x, y)]['completed'] += len(completed)
            self.task_stats['uav'][(x, y)]['successful'] += success

            # === UAV maintenance ===
            uav.update_cache(
                self.g_timestep,
                global_satellite_content_pool=self.global_satellite_content_pool,
                is_connected_to_satellite=(x, y) in self.connected_uavs
            )
            uav.clear_aggregated_content()

            # === Energy usage ===
            #uav.energy -= 150 * self.duration  # hovering + caching cost
            #uav.energy_used_this_slot += 150 * self.duration

            if uav.energy <= 0:
                print(f"[CRITICAL] UAV at ({x}, {y}) has drained all energy at {self.g_timestep}. SYSTEM DOWN.")
                raise SystemDownException(f"[CRITICAL] UAV at ({x}, {y}) has drained all energy. SYSTEM DOWN.")

        # === Satellite executes tasks ===
        for sat in self.sats:
            completed = sat.execute_tasks(self.g_timestep, self.global_satellite_content_pool)
            success = sum(
                1 for t in completed
                if t.get('delay', {}).get('total_delay', float('inf')) <= t.get('delay_bound', float('inf'))
            )

            if sat.sat_id not in self.task_stats['satellite']:
                self.task_stats['satellite'][sat.sat_id] = {'received': 0, 'completed': 0, 'successful': 0}

            self.task_stats['satellite'][sat.sat_id]['completed'] += len(completed)
            self.task_stats['satellite'][sat.sat_id]['successful'] += success

            # === Satellite eviction ===
        self.evict_expired_content()
