# communication_model.py - COMPLETE REPLACEMENT
import numpy as np


class CommunicationModel:
    """
    Complete implementation of SAGIN communication model as per paper equations
    Replaces the old compute_rate_general function
    """

    def __init__(self):
        # Physical constants
        self.c = 3e8  # Speed of light (m/s)

        # Frequency bands as per paper
        self.C_BAND_FREQ = 6e9  # 6 GHz (C-band center for IoT-UAV)
        self.KA_BAND_FREQ = 30e9  # 30 GHz (Ka-band center for UAV-Satellite)

        # Bandwidth allocations as per paper
        self.B_CU = 5e6  # C-band IoT-UAV total bandwidth (5 MHz)
        self.B_US = 10e6  # Ka-band UAV-Satellite per subchannel (10 MHz)

        # Power specifications (Watts)
        self.P_IOT = 0.1  # IoT device transmit power
        self.P_UAV = 10.0  # UAV transmit power to satellite
        self.P_SAT = 100.0  # Satellite transmit power

        # Antenna gains (linear scale, not dB)
        self.G_IOT = 1.0  # IoT omnidirectional antenna
        self.G_UAV_RX = 10.0  # UAV receive antenna (10 dBi = 10x)
        self.G_UAV_TX = 100.0  # UAV transmit antenna (20 dBi = 100x)
        self.G_SAT = 10000.0  # Satellite high-gain antenna (40 dBi = 10000x)

        # Noise power (Watts)
        self.NOISE_C_BAND = 1e-12  # C-band thermal noise
        self.NOISE_KA_BAND = 1e-13  # Ka-band thermal noise

    def compute_path_loss(self, distance, frequency, path_loss_exponent=2.0):
        """
        Free-space path loss: L = (4πfd/c)^n
        """
        if distance <= 0:
            distance = 1.0  # Prevent division by zero
        wavelength = self.c / frequency
        path_loss = (4 * np.pi * distance / wavelength) ** path_loss_exponent
        return path_loss

    def compute_iot_to_uav_rate(self, iot_pos, uav_pos, interference=0.0, fading=1.0):
        """
        IoT-to-UAV communication using C-band with TDMA
        Implements Paper Equation (5): R^(m)_{x,y}(t) = B_CU * log2(1 + SINR)

        Returns:
        - rate (bps): Data rate in bits per second
        - success (bool): Whether communication is feasible
        - delay_func: Function to calculate transmission delay for given content size
        """
        # Calculate 3D distance
        distance = np.linalg.norm(np.array(iot_pos) - np.array(uav_pos))

        # Path loss calculation
        path_loss = self.compute_path_loss(distance, self.C_BAND_FREQ)

        # Channel gain with fading: G^(m)_{x,y}(t) = fading * G_tx * G_rx / L
        channel_gain = fading * self.G_IOT * self.G_UAV_RX / path_loss

        # Signal power
        signal_power = self.P_IOT * channel_gain

        # SINR calculation (Equation 5): SINR = P_m * G^(m)_{x,y}(t) / (I_{x,y}(t) + σ²)
        noise_plus_interference = self.NOISE_C_BAND + interference
        sinr = signal_power / noise_plus_interference

        # Data rate: R^(m)_{x,y}(t) = B_CU * log2(1 + SINR)
        rate = self.B_CU * np.log2(1 + sinr)

        # Success if rate is above threshold (100 kbps minimum)
        success = rate >= 1e5

        # Transmission delay function for content of given size (MB)
        def transmission_delay_func(content_size_mb):
            if rate <= 0:
                return float('inf')
            content_bits = content_size_mb * 8 * 1e6  # Convert MB to bits
            return content_bits / rate  # Time = Data / Rate

        return rate, success, transmission_delay_func

    def compute_uav_to_satellite_uplink_rate(self, uav_pos, sat_pos, subchannel_assigned=True, fading=1.0):
        """
        UAV-to-Satellite uplink using Ka-band with OFDMA
        Implements Paper Equation (8): R^up_{x,y,k}(t) = B_US * log2(1 + SINR)
        """
        if not subchannel_assigned:
            return 0.0, False, lambda x: float('inf')

        # Calculate distance
        distance = np.linalg.norm(np.array(uav_pos) - np.array(sat_pos))

        # Path loss
        path_loss = self.compute_path_loss(distance, self.KA_BAND_FREQ)

        # Channel gain: G_{x,y,k}(t)
        channel_gain = fading * self.G_UAV_TX * self.G_SAT / path_loss

        # Signal power
        signal_power = self.P_UAV * channel_gain

        # SINR (no interference assumed in Ka-band)
        sinr = signal_power / self.NOISE_KA_BAND

        # Uplink rate: R^up_{x,y,k}(t) = B_US * log2(1 + SINR)
        rate = self.B_US * np.log2(1 + sinr)

        # Propagation delay (Equation 10): τ^US_{x,y,k} = ||p_{ux,y} - p_{sk}|| / c
        propagation_delay = distance / self.c

        success = rate >= 1e6  # 1 Mbps minimum for satellite links

        def transmission_delay_func(content_size_mb):
            if rate <= 0:
                return float('inf')
            content_bits = content_size_mb * 8 * 1e6
            transmission_time = content_bits / rate
            return transmission_time + propagation_delay  # Total delay

        return rate, success, transmission_delay_func

    def compute_satellite_to_uav_downlink_rate(self, sat_pos, uav_pos, subchannel_assigned=True, fading=1.0):
        """
        Satellite-to-UAV downlink using Ka-band with OFDMA
        Implements Paper Equation (9): R^down_{x,y,k}(t) = B_US * log2(1 + SINR)
        """
        if not subchannel_assigned:
            return 0.0, False, lambda x: float('inf')

        # Calculate distance
        distance = np.linalg.norm(np.array(sat_pos) - np.array(uav_pos))

        # Path loss
        path_loss = self.compute_path_loss(distance, self.KA_BAND_FREQ)

        # Channel gain: G_{k,x,y}(t)
        channel_gain = fading * self.G_SAT * self.G_UAV_RX / path_loss

        # Signal power
        signal_power = self.P_SAT * channel_gain

        # SINR
        sinr = signal_power / self.NOISE_KA_BAND

        # Downlink rate: R^down_{x,y,k}(t) = B_US * log2(1 + SINR)
        rate = self.B_US * np.log2(1 + sinr)

        # Propagation delay
        propagation_delay = distance / self.c

        success = rate >= 1e6

        def transmission_delay_func(content_size_mb):
            if rate <= 0:
                return float('inf')
            content_bits = content_size_mb * 8 * 1e6
            transmission_time = content_bits / rate
            return transmission_time + propagation_delay

        return rate, success, transmission_delay_func

    def estimate_co_channel_interference(self, victim_uav_pos, interfering_regions, region_size=(100, 100)):
        """
        Estimate co-channel interference I_{x,y}(t) from other IoT regions
        Used in Equation (5) for IoT-to-UAV communication
        """
        total_interference = 0.0

        for region_coord in interfering_regions:
            region_x, region_y = region_coord
            # Assume one active IoT transmitter per region at region center
            interferer_pos = (
                region_x * region_size[0] + region_size[0] / 2,
                region_y * region_size[1] + region_size[1] / 2,
                0  # Ground level
            )

            # Distance from interferer to victim UAV
            distance = np.linalg.norm(np.array(interferer_pos) - np.array(victim_uav_pos))
            if distance <= 0:
                continue

            # Path loss for interference signal
            path_loss = self.compute_path_loss(distance, self.C_BAND_FREQ)

            # Interference power (same transmit power as useful signal)
            interference_power = self.P_IOT * self.G_IOT / path_loss
            total_interference += interference_power

        return total_interference

    def check_tdma_feasibility(self, content_sizes, slot_duration):
        """
        Check if all content can be transmitted within TDMA slot duration
        Used for IoT-to-UAV transmission scheduling
        """
        # Use conservative rate estimate (worst case SINR = 1)
        min_rate = self.B_CU * np.log2(1 + 1)

        total_transmission_time = 0
        for size_mb in content_sizes:
            content_bits = size_mb * 8 * 1e6
            transmission_time = content_bits / min_rate
            total_transmission_time += transmission_time

        return total_transmission_time <= slot_duration


# Backward compatibility function for existing code
def compute_rate_general(sender_pos, receiver_pos, **kwargs):
    """
    Backward compatibility wrapper for existing code
    This will be gradually replaced with specific communication methods
    """
    # Create communication model instance
    comm_model = CommunicationModel()

    # Default to IoT-to-UAV communication for compatibility
    rate, success, delay_func = comm_model.compute_iot_to_uav_rate(
        sender_pos, receiver_pos,
        interference=kwargs.get('interference', 0.0),
        fading=kwargs.get('fading', 1.0)
    )

    return rate, success


# Testing and validation
if __name__ == "__main__":
    print("=== Testing Communication Model ===")
    comm_model = CommunicationModel()

    # Test IoT-to-UAV communication
    iot_pos = (10, 10, 0)  # Ground level IoT device
    uav_pos = (50, 50, 100)  # UAV at 100m altitude

    print("\n1. IoT-to-UAV Communication (C-band, TDMA):")
    rate, success, delay_func = comm_model.compute_iot_to_uav_rate(iot_pos, uav_pos)
    print(f"   Rate: {rate / 1e6:.2f} Mbps")
    print(f"   Success: {success}")
    print(f"   Transmission delay for 5MB: {delay_func(5.0):.3f}s")

    # Test with interference
    interfering_regions = [(0, 0), (1, 1), (2, 2)]
    interference = comm_model.estimate_co_channel_interference(uav_pos, interfering_regions)
    rate_with_interference, _, delay_func_interference = comm_model.compute_iot_to_uav_rate(
        iot_pos, uav_pos, interference=interference
    )
    print(f"   Rate with interference: {rate_with_interference / 1e6:.2f} Mbps")
    print(f"   Interference power: {interference:.2e} W")

    # Test UAV-to-Satellite communication
    sat_pos = (100, 500, 550000)  # LEO satellite

    print("\n2. UAV-to-Satellite Uplink (Ka-band, OFDMA):")
    uplink_rate, up_success, up_delay_func = comm_model.compute_uav_to_satellite_uplink_rate(
        uav_pos, sat_pos, subchannel_assigned=True
    )
    print(f"   Rate: {uplink_rate / 1e6:.2f} Mbps")
    print(f"   Success: {up_success}")
    print(f"   Upload delay for 5MB: {up_delay_func(5.0):.3f}s")

    print("\n3. Satellite-to-UAV Downlink (Ka-band, OFDMA):")
    downlink_rate, down_success, down_delay_func = comm_model.compute_satellite_to_uav_downlink_rate(
        sat_pos, uav_pos, subchannel_assigned=True
    )
    print(f"   Rate: {downlink_rate / 1e6:.2f} Mbps")
    print(f"   Success: {down_success}")
    print(f"   Download delay for 5MB: {down_delay_func(5.0):.3f}s")

    # Test TDMA feasibility
    print("\n4. TDMA Feasibility Check:")
    content_sizes = [2.0, 5.0, 8.0, 12.0]  # MB
    slot_duration = 300  # seconds
    feasible = comm_model.check_tdma_feasibility(content_sizes, slot_duration)
    print(f"   Content sizes: {content_sizes} MB")
    print(f"   Slot duration: {slot_duration}s")
    print(f"   TDMA feasible: {feasible}")

    print("\n=== Communication Model Test Complete ===")