# iot_region.py - COMPLETE REPLACEMENT
import numpy as np
from communication_model import CommunicationModel


class IoTRegion:
    def __init__(self, num_iot_devices, max_active_iot, duration, region_coords, region_size=(100, 100)):
        self.num_iot_devices = num_iot_devices
        self.max_active_device = max_active_iot
        self.region_coords = region_coords
        self.duration = duration
        self.region_size = region_size

        # Initialize communication model
        self.comm_model = CommunicationModel()

        # Generate IoT device positions within region boundaries
        self.iot_positions = []
        region_x, region_y = region_coords
        width, height = region_size

        for device_id in range(num_iot_devices):
            # Random position within region
            xi = region_x * width + np.random.uniform(0, width)
            yi = region_y * height + np.random.uniform(0, height)
            zi = 0  # IoT devices at ground level
            self.iot_positions.append((xi, yi, zi))

        # Zipf parameters for spatiotemporal variation (Paper Equation 1)
        self.a_base = np.random.uniform(1.2, 2.5)  # Base Zipf parameter
        self.timestep = 0
        self.current_zipf_param = self.a_base

    def update_zipf_param(self):
        """
        Update Zipf parameter with spatiotemporal variation
        Implements Paper Equation (1): α_{x,y}(t) = a_base + φ_spatial(x,y) + φ_temporal(t,x,y)
        """
        x, y = self.region_coords

        # Spatial component: φ_spatial(x,y) = ((3x + 2y) mod 5) × 0.3
        phi_spatial = ((x * 3 + y * 2) % 5) * 0.3

        # Temporal component: φ_temporal(t,x,y) = 0.6 × sin(2πt/T + x + y)
        # Using timestep as proxy for t, and assuming reasonable period
        phi_temporal = 0.6 * np.sin((2 * np.pi * self.timestep / 20) + x + y)

        # Final Zipf shape parameter
        self.current_zipf_param = self.a_base + phi_spatial + phi_temporal
        self.timestep += 1

        # Store for debugging/analysis
        self.a_t = self.current_zipf_param

    def sample_active_devices(self):
        """
        Sample active IoT devices using spatiotemporal Zipf distribution
        Implements Paper Equations (2) and (3)
        """
        # Update Zipf parameter first
        self.update_zipf_param()

        # Number of active devices (Paper Equation 3)
        # A^x,y_num(t) = min(κ × α_{x,y}(t) × N^x,y_total, N^x,y_total)
        kappa = 0.8  # Scaling factor κ ∈ (0,1]
        num_active = min(
            int(kappa * self.current_zipf_param * self.num_iot_devices),
            self.max_active_device,
            self.num_iot_devices
        )

        if num_active <= 0:
            return []

        # Zipf probability distribution (Paper Equation 2)
        # Pr(m)_{x,y}(t) = m^{-α_{x,y}(t)} / Σ_{n=1}^N n^{-α_{x,y}(t)}
        ranks = np.arange(1, self.num_iot_devices + 1)
        zipf_probs = np.power(ranks, -self.current_zipf_param)
        zipf_probs = zipf_probs / zipf_probs.sum()  # Normalize to sum to 1

        # Sample devices according to Zipf distribution without replacement
        try:
            selected_devices = np.random.choice(
                self.num_iot_devices,
                size=num_active,
                replace=False,
                p=zipf_probs
            )
            return selected_devices.tolist()
        except ValueError:
            # Fallback in case of numerical issues
            return np.random.choice(self.num_iot_devices, size=num_active, replace=False).tolist()

    def generate_content(self, active_device_ids, timestep, grid_coord=None):
        """
        Generate content from active IoT devices
        Implements Paper Equation (4): c^(m)_{x,y}(t) = ((x,y,m), ε^(m)_{x,y}, TTL^(m)_{x,y}, t×δ)
        """
        content_items = []
        region_x, region_y = self.region_coords

        for device_id in active_device_ids:
            # Content ID: (x, y, m) where m is device ID
            if grid_coord is not None:
                content_id = (grid_coord[0], grid_coord[1], device_id)
            else:
                content_id = (region_x, region_y, device_id)

            # Get IoT device position
            if device_id < len(self.iot_positions):
                iot_pos = self.iot_positions[device_id]
            else:
                # Fallback for edge case
                iot_pos = (region_x * self.region_size[0] + 50,
                           region_y * self.region_size[1] + 50, 0)

            # Content metadata following Paper Equation (4)
            content = {
                'id': content_id,
                'size': np.random.uniform(1, 20),  # ε^(m)_{x,y} ~ U(1, 20) MB
                'ttl': np.random.randint(600, 1800),  # TTL^(m)_{x,y} ~ U(600, 1800) seconds
                'generation_time': timestep * self.duration,  # t × δ (slot time)
                'popularity': 0,  # Will be updated based on usage
                'iot_pos': iot_pos,  # Physical position for communication calculation
                'device_id': device_id,  # For tracking purposes
                'region_coords': self.region_coords,  # Source region
                'received_by_uav': 0,  # Will be set during aggregation
                'received_by_satellite': 0,  # Will be set during upload
            }

            content_items.append(content)

        return content_items

    def compute_iot_to_uav_rate(self, iot_pos, uav_pos, interference=0.0, G_tx=1.0, G_rx=10.0):
        """
        Compute IoT-to-UAV communication rate using proper communication model
        Replaces the old generic rate calculation with paper-specific implementation

        Returns:
        - rate: Data rate in bps
        - success: Boolean indicating if communication is feasible
        """
        # Use the proper communication model instead of generic calculation
        rate, success, delay_func = self.comm_model.compute_iot_to_uav_rate(
            iot_pos=iot_pos,
            uav_pos=uav_pos,
            interference=interference,
            fading=1.0  # Assume good channel conditions by default
        )

        # Return rate and success for backward compatibility
        return rate, success

    def check_tdma_slot_feasibility(self, active_devices_content, slot_duration):
        """
        Check if all active device content can be transmitted within TDMA slot
        This ensures TDMA protocol constraints are met
        """
        content_sizes = [content['size'] for content in active_devices_content]
        return self.comm_model.check_tdma_feasibility(content_sizes, slot_duration)

    def get_region_statistics(self):
        """
        Get current region statistics for monitoring and debugging
        """
        return {
            'region_coords': self.region_coords,
            'num_iot_devices': self.num_iot_devices,
            'max_active_devices': self.max_active_device,
            'current_zipf_param': getattr(self, 'current_zipf_param', self.a_base),
            'base_zipf_param': self.a_base,
            'timestep': self.timestep,
            'region_size': self.region_size
        }

    def estimate_interference_to_uav(self, target_uav_pos, other_regions):
        """
        Estimate interference from other regions to a target UAV
        Used for co-channel interference calculation
        """
        return self.comm_model.estimate_co_channel_interference(
            target_uav_pos, other_regions, self.region_size
        )

    def get_content_transmission_order(self, content_items, uav_pos):
        """
        Determine optimal transmission order for TDMA scheduling
        Prioritizes smaller content for efficient slot utilization
        """
        # Calculate transmission times for each content
        content_with_times = []
        for content in content_items:
            iot_pos = content['iot_pos']
            rate, success, delay_func = self.comm_model.compute_iot_to_uav_rate(
                iot_pos, uav_pos, interference=0.0
            )

            if success:
                transmission_time = delay_func(content['size'])
                content_with_times.append((content, transmission_time))

        # Sort by transmission time (shortest first for TDMA efficiency)
        content_with_times.sort(key=lambda x: x[1])

        return [item[0] for item in content_with_times]

    def reset_timestep(self):
        """
        Reset timestep counter (useful for new episodes in RL training)
        """
        self.timestep = 0
        self.current_zipf_param = self.a_base


# Testing and validation
if __name__ == "__main__":
    print("=== Testing IoT Region ===")

    # Create test region
    region = IoTRegion(
        num_iot_devices=20,
        max_active_iot=10,
        duration=300,
        region_coords=(1, 2),
        region_size=(100, 100)
    )

    print(f"Region statistics: {region.get_region_statistics()}")

    # Test multiple timesteps to see spatiotemporal variation
    print("\nTesting spatiotemporal variation:")
    for t in range(5):
        active_devices = region.sample_active_devices()
        content_items = region.generate_content(active_devices, t)

        print(f"Timestep {t}:")
        print(f"  Zipf parameter: {region.current_zipf_param:.3f}")
        print(f"  Active devices: {len(active_devices)} ({active_devices})")
        print(f"  Generated content: {len(content_items)} items")
        content_sizes = [f"{c['size']:.1f}MB" for c in content_items]
        print(f"  Content sizes: {content_sizes}")

    # Test communication rate calculation
    print(f"\nTesting communication rates:")
    uav_pos = (150, 250, 100)  # UAV position
    iot_pos = (120, 220, 0)  # IoT device position

    rate, success = region.compute_iot_to_uav_rate(iot_pos, uav_pos)
    print(f"IoT-to-UAV rate: {rate / 1e6:.2f} Mbps, Success: {success}")

    # Test with interference
    other_regions = [(0, 0), (2, 2), (1, 1)]
    interference = region.estimate_interference_to_uav(uav_pos, other_regions)
    rate_with_int, success_with_int = region.compute_iot_to_uav_rate(
        iot_pos, uav_pos, interference=interference
    )
    print(f"With interference: {rate_with_int / 1e6:.2f} Mbps, Success: {success_with_int}")

    # Test TDMA feasibility
    test_content = region.generate_content([0, 1, 2, 3], 0)
    feasible = region.check_tdma_slot_feasibility(test_content, 300)
    print(f"TDMA feasible for {len(test_content)} items: {feasible}")

    print("\n=== IoT Region Test Complete ===")