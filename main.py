# main.py (SEQUENCED SIMULATION FLOW WITH TASK + CACHE DECISION + GRU-PPO)
import numpy as np
from communication_model import compute_rate
from sagin_env import SAGINEnv
from ppo_gru_agent import GRUPPOAgent

X, Y = 3, 3
num_contents = 10
cache_size = 10
max_queue = 10
num_sats = 2
num_iot_per_region = 15
max_active_iot=8
ofdm_slots = 9
obs_dim = 28 # example dim for observation vector, must match UAV.observe output

# Initialize environment and PPO agent
env = SAGINEnv(X, Y, num_contents, cache_size, max_queue, num_sats, num_iot_per_region, max_active_iot, ofdm_slots)
ppo_agent = GRUPPOAgent(obs_dim=obs_dim, num_contents=num_contents)

# Run simulation
for timestep in range(500):
    print(f"\n=== Time Slot {timestep + 1} ===")

    # (1) IoT activation and data aggregation
    env.collect_iot_data(timestep, compute_rate)

    # (2) Allocate OFDM slots
    env.allocate_ofdm_slots()

    # (2.5) Update satellite coverage
    for sat in env.sats:
        sat.update_coverage(timestep)

    # (3) Only UAVs with slots upload content to satellites
    env.upload_to_satellites()

    # (4) Sync content across satellites
    env.sync_satellites()

    # (5) Generate tasks and make offloading decisions
    env.generate_and_offload_tasks()

    # (6) UAV-local caching based on PPO agent decisions
    actions = {}
    for (x, y), uav in env.uavs.items():
        uav.current_time = timestep
        obs = uav.observe(
            neighbor_loads=np.zeros((4, 1)),  # Simplified for test run
            satellite_in_range=[1] * len(env.sats),
            activation_mask=np.ones(num_contents)
        )
        cache_action, offload_action, _ = ppo_agent.act(obs)
        actions[(x, y)] = {'cache': cache_action, 'offload': offload_action}

    env.step(actions, timestep)

# Log summary
print("\n=== Last 10 Task Logs ===")
for entry in env.task_log[-10:]:
    print("Task Log Entry:", entry)

# Log success summary
print("\n=== Task Success Summary ===")
for entry in env.success_log:
    print(f"UAV {entry['coord']} completed {entry['completed']} tasks, "
          f"of which {entry['successful']} met delay bound.")

# === Task Success Summary ===
total_completed = sum(e['completed'] for e in env.success_log)
total_successful = sum(e['successful'] for e in env.success_log)
success_rate = 100 * total_successful / total_completed if total_completed else 0
print(f"Overall Task Success Rate: {success_rate:.2f}%")

print("Per-UAV Task Performance:")
for e in env.success_log:
    rate = 100 * e['successful'] / e['completed'] if e['completed'] else 0
    print(f"UAV {e['coord']} — Completed: {e['completed']}, Within Deadline: {e['successful']} ({rate:.1f}%)")

print("Cache Hit Rates per UAV:")
for (x, y), uav in env.uavs.items():
    if uav.total_tasks > 0:
        hit_rate = 100 * uav.cache_hits / uav.total_tasks
        print(f"UAV ({x},{y}) — Cache Hit Rate: {hit_rate:.1f}%")

print(f"Total Dropped Tasks: {env.dropped_tasks}")
