# test_enhanced_task_generation.py
from sagin_env import SAGINEnv


def test_enhanced_task_generation():
    # Create small test environment
    env = SAGINEnv(X=3, Y=3, duration=300, cache_size=40,
                   compute_power_uav=25, compute_power_sat=200,
                   energy=80000, max_queue=15, num_sats=2,
                   num_iot_per_region=20, max_active_iot=10, ofdm_slots=6)

    print("🧪 Testing Enhanced Task Generation")
    print("=" * 50)

    # Test for several timesteps
    for timestep in range(12):  # Test across 2+ epochs
        print(f"\n--- Timestep {timestep} ---")

        # Generate tasks for each UAV
        for (x, y), uav in env.uavs.items():
            tasks = uav.generate_tasks(env.X, env.Y, timestep, num_tasks=4)

            # Print task analysis
            if tasks:
                own_region = sum(1 for t in tasks if t['content_id'][:2] == (x, y))
                cross_region = len(tasks) - own_region
                epoch = tasks[0]['epoch']

                print(f"  UAV {(x, y)}: {len(tasks)} tasks (epoch {epoch}) "
                      f"- Own: {own_region}, Cross: {cross_region}")

    # Print final statistics
    print("\n📊 FINAL STATISTICS")
    print("=" * 50)
    for (x, y), uav in env.uavs.items():
        stats = uav.get_task_generation_statistics()
        print(f"UAV {(x, y)}: {stats}")


if __name__ == "__main__":
    test_enhanced_task_generation()