# SAGIN:  Content Caching and Task Offloading Framework in Space-Air-Ground Integrated Networks

This project implements a **Deep Reinforcement Learning-based framework** for **joint content caching and task offloading** in **SAGIN (Space-Air-Ground Integrated Network)** environments. It simulates UAVs (Unmanned Aerial Vehicles), satellites, and ground-based IoT devices collaborating for efficient data processing, content caching, and task execution under resource constraints.

---

## 🚀 Overview of SAGIN Architecture

```
                 +----------------------------+
                 |        Satellite Tier      |
                 |   (Global Coverage, High   |
                 |     Storage & Computing)   |
                 +----------------------------+
                            ↑   ↑   ↑
                            ↓   ↓   ↓
       +---------------------------------------------+
       |              Aerial Tier (UAVs)             |
       |    (Mobile Nodes, Cache/Compute Enabled)    |
       +---------------------------------------------+
            ↑     ↑       ↑       ↑        ↑
            ↓     ↓       ↓       ↓        ↓
+------------------------------------------------------------+
|           Ground Tier (IoT Regions + Sensors)              |
|   (Local Data Generation, Periodic Activation, Content ID) |
+------------------------------------------------------------+
```

- **IoT Devices** generate content and tasks.
- **UAVs** aggregate, cache, and process tasks or offload them to neighbors or satellites.
- **Satellites** act as powerful compute/cache nodes with global coverage.

---

## ⚙️ Code Structure and Modules

| File                          | Purpose |
|------------------------------|---------|
| `main.py`                    | Runs PPO-driven simulation (learning agent) |
| `compare_policies.py`        | Baseline simulation (popularity-based caching, greedy offloading) |
| `sagin_env.py`               | Core environment managing UAVs, satellites, IoT regions |
| `uav.py`                     | UAV class: caching, task execution, energy handling |
| `satellite.py`               | Satellite class: task queueing, execution, TTL-based eviction |
| `iot_region.py`              | Models IoT regions, activation, content generation |
| `ppo_gru_agent.py`           | PPO + GRU agent for intelligent caching decisions |
| `content_popularity_predictor.py` | GRU-based popularity predictor (for future use) |
| `communication_model.py`     | Channel modeling between UAVs and IoT devices |

---

## 🔄 Simulation Flow

1. **IoT Device Activation**  
   At each time step, IoT devices generate content based on Zipf-distributed popularity.

2. **UAV Aggregation**  
   UAVs receive content, transmit to satellites (if connected), and consume energy.

3. **OFDM Slot Allocation**  
   UAVs get randomly assigned slots to communicate with satellites.

4. **Satellite Caching**  
   Satellites receive and store content with TTL constraints.

5. **Task Generation**  
   UAVs generate computational tasks that need associated content.

6. **Offloading Decisions**  
   - First try local cache  
   - Then neighbor UAVs  
   - Then satellite  
   - If not found, task is dropped

7. **Task Execution**  
   UAVs and satellites execute queued tasks if delays are within bounds.

8. **Caching Decision via PPO**  
   A GRU-based PPO agent decides which content to cache in UAVs based on observed energy, content popularity, neighbor loads, etc.

9. **Eviction & Energy Update**  
   Expired content is evicted. UAV energy is updated and checked—simulation halts if any UAV's energy is depleted.

---

## ▶️ How to Run

### A. Baseline Policy (no RL):
```bash
python compare_policies.py
```
This runs the simulation using:
- Popularity-based caching
- Greedy offloading
- 5 episodes of 5 time slots each

### B. PPO Policy (learning-based):
```bash
python main.py
```
This:
- Loads a PPO+GRU agent
- Executes 500 time slots
- Learns caching decisions per UAV

---

## 📦 Dependencies

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib

Install with:
```bash
pip install numpy torch matplotlib
```

---

## 📊 Output

- Task logs and success summaries printed after each run
- Cache hit rates and delay-bound statistics per UAV
- Optional plots in `compare_policies.py`

---

## ⚠️ Notes

- The simulation stops **immediately** if **any UAV's energy becomes zero**, simulating real-world failure.
- Satellites have **universal coverage**; they do not move.
- UAVs cache content within limited memory and update their cache per time slot.
- Content and tasks have TTLs and delay bounds.

---

## 🧠 Future Extensions

- Multi-agent PPO
- Federated caching across UAVs
- Energy-aware path planning for UAV movement
- Satellite handovers and mobility

---

## 👨‍💻 Authors

This codebase was developed as part of research in **Intelligent Communication Networks** and **AI-Driven SAGIN Optimization**. For questions or collaboration, reach out to the project maintainer.

---