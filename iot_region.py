# iot_region.py (UPDATED TO HANDLE compute_rate_func INPUT)
import numpy as np
from communication_model import compute_rate_general

import time

class IoTRegion:
    def __init__(self, num_iot_devices,  max_active_iot, duration, region_coords, region_size=(100, 100)):
        self.num_iot_devices = num_iot_devices
        self.max_active_device = max_active_iot
        self.region_coords = region_coords

        self.iot_positions = []
        region_x, region_y = region_coords
        width, height = region_size
        for _ in range(num_iot_devices):
            xi = region_x * width + np.random.uniform(0, width)
            yi = region_y * height + np.random.uniform(0, height)
            self.iot_positions.append((xi, yi))

        self.a_base = np.random.uniform(1.2, 2.5)
        self.timestep = 0
        self.duration= duration

    def compute_iot_to_uav_rate(self, iot_pos, uav_pos):
        return compute_rate_general(
            sender_pos=iot_pos + (0,),  # IoT ground node (z = 0)
            receiver_pos=uav_pos,  # UAV position (x, y, z)
            bandwidth=5e6,  # 5 MHz
            P_tx=0.1,  # IoT device transmit power
            fc=2e9,  # 2 GHz
            G_tx=1,  # isotropic
            G_rx=10,  # moderate UAV gain
            noise=1e-9,
            fading=1.0
        )

    def update_zipf_param(self):
            x, y = self.region_coords
            # Stronger spatial bias: different base pattern per region
            spatial_bias = ((x * 3 + y * 2) % 5) * 0.3  # range: [0, 1.2]

            # Temporal component with phase offset by region
            temporal_bias = 0.6 * np.sin((self.timestep / 6) + (x + y))  # range: [-0.6, +0.6]

            # Final Zipf shape parameter
            self.a_t = self.a_base + spatial_bias + temporal_bias
            self.timestep += 1

    def sample_active_devices(self): #devices are getting active based on regional zipf value
        self.update_zipf_param()
        ranks = np.arange(1, self.num_iot_devices + 1)
        zipf_probs = 1 / np.power(ranks, self.a_t)
        zipf_probs /= zipf_probs.sum()
        selected = np.random.choice(self.num_iot_devices, size=min(self.max_active_device, self.num_iot_devices), replace=False, p=zipf_probs)
        return selected.tolist()

    def generate_content(self, active_device_ids, timestep, grid_coord=None):
        content_items = []
        region_x, region_y = self.region_coords

        for device_type_id in active_device_ids:
            # Content ID is always (grid_x, grid_y, device_type_id)
            if grid_coord is not None:
                content_id = (grid_coord[0], grid_coord[1], device_type_id)
            else:
                content_id = (region_x, region_y, device_type_id)
            # Get the IoT device position for this device_id
            iot_pos = self.iot_positions[device_type_id]

            content = {
                'id': content_id,
                'size': np.random.uniform(1, 20),  # MB
                'ttl': np.random.randint(600, 1800),  # seconds
                'popularity': 0,
                'generation_time': timestep * self.duration,
                'received_by_uav': 0,
                'received_by_satellite': 0,
                'iot_pos': iot_pos,  # <--- This is important!
            }

            content_items.append(content)

        return content_items

