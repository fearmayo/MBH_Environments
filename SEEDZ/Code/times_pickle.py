import pickle
import numpy as np

path = "/home/daxal/data/ProductionRuns/Renaissance/NoFeedback/sink_particle.pkl"

with open(path, "rb") as f:
    sinks = pickle.load(f)

all_times = []

print(f"Loaded {len(sinks)} sinks")

for sid, sd in sinks.items():
    evo = sd.get("evolution", {})

    # Collect all times for this sink
    times = list(evo.keys())
    all_times.extend(times)

    # Print a sample for a few sinks
    print(f"\nSink {sid}: {len(times)} evolution entries")
    print(f"First 5 times: {sorted(times)[:5]}")
    print(f"Last 5 times: {sorted(times)[-5:]}")

# Now summarise across all sinks
unique_times = sorted(set(all_times))

print("\n===================================")
print("Summary of all evolution times in the pkl:")
print(f"Number of unique time values: {len(unique_times)}")
print(f"Min time: {min(unique_times)}")
print(f"Max time: {max(unique_times)}")
print("Expected a(z) range for z≈20 → z≈10 :", "0.047 → 0.091")
print("Unique times:")
for t in unique_times:
    print(t)
