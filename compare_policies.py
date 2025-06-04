# compare_policies_clean_baseline.py
import numpy as np
import matplotlib.pyplot as plt
from sagin_env import SAGINEnv
from sagin_env import SystemDownException

# from ppo_gru_agent import GRUPPOAgent


from sagin_env import SystemDownException

def run_policy(env, episodes=5):
    logs = {
        'success_log': [], 'energy': [], 'cache_hits': [], 'task_log': [], 'dropped': 0
    }

    try:
        for ep in range(episodes):
            print('episode:', ep)
            for timestep in range(50):
                env.collect_iot_data()
                env.allocate_ofdm_slots()
                for sat in env.sats:
                    sat.update_coverage(timestep)
                env.upload_to_satellites()
                env.sync_satellites()
                env.generate_and_offload_tasks()
                env.step()  # <- may raise SystemDownException

    except SystemDownException as e:
        print(e)
        print("Simulation ended due to energy depletion.")

    logs['success_log'] = env.success_log
    logs['task_log'] = env.task_log
    logs['dropped'] = env.dropped_tasks
    return logs



def summarize_and_plot(baseline_log):
    def get_success_rate(log):
        total = sum(e['completed'] for e in log)
        ok = sum(e['successful'] for e in log)
        return 100 * ok / total if total else 0

    labels = ['Success Rate (%)', 'Avg Cache Hit', 'Avg Energy Used', 'Dropped Tasks']
    baseline_vals = [
        get_success_rate(baseline_log['success_log']),
        np.mean(baseline_log['cache_hits']),
        np.mean(baseline_log['energy']),
        baseline_log['dropped']
    ]

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar(x, baseline_vals, width, label='Baseline (Pop-based Caching)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Baseline Policy Performance")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, Y = 10, 10
    cache_size = 2048
    compute_power_uav = 30
    compute_power_sat=200
    energy = 540000
    max_queue = 10
    num_sats = 2
    num_iot_per_region = 50
    max_active_iot = 25
    ofdm_slots = 9
    duration = 300

    env = SAGINEnv(X, Y, duration, cache_size, compute_power_uav, compute_power_sat, energy,
                   max_queue, num_sats, num_iot_per_region, max_active_iot, ofdm_slots)

    logs_baseline = run_policy(env, episodes=1)
    print("\n=== UAV Task Stats ===")
    for coord, stats in env.task_stats['uav'].items():
        print(
            f"UAV {coord}: Generated={stats['generated']}, Completed={stats['completed']}, Successful={stats['successful']}")

    print("\n=== Satellite Task Stats ===")
    for sid, stats in env.task_stats['satellite'].items():
        print(
            f"Satellite {sid}: Received={stats['received']}, Completed={stats['completed']}, Successful={stats['successful']}")

    #summarize_and_plot(logs_baseline)

    # === Ready for PPO Extension Later ===
    # ppo_agent = GRUPPOAgent(obs_dim=..., num_contents=...)
    # logs_ppo = run_policy(env, agent=ppo_agent, use_ppo=True)
    # summarize_and_plot(logs_baseline, logs_ppo)
