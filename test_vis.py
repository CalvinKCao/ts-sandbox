import numpy as np

# Simulate some data
L = 20
H = 16
max_scale = 3.5

# Noise: some specific values with negatives
noise = np.array([0.5, -1.2, 0.8, -0.3, -2.1, 1.5, 0.0, -0.9, 2.5, -1.5,
                  0.3, -0.1, -0.5, 1.1, -0.2, 0.4, -1.8, 0.7, -0.6, 1.2])

# The pipeline standardizes the noise first!
noise_mean = noise.mean()
noise_std = noise.std() + 1e-8
noise_norm = (noise - noise_mean) / noise_std

# Create 2D map (CDF occupancy)
image = np.zeros((H, L), dtype=int)
for i, x in enumerate(noise_norm):
    # bin index logic from the paper/code
    b = int(np.clip(np.floor((x + max_scale) / (2 * max_scale) * H), 0, H - 1))
    image[:b+1, i] = 1

print("Normalized Noise Values:")
for i, x in enumerate(noise_norm):
    print(f"t={i:02d}: {x:+5.2f}", end="  ")
    if (i+1) % 5 == 0: print()

print("\n2D Occupancy Map (Y-axis = value bins, X-axis = time):")
for r in range(H-1, -1, -1):
    row_str = ""
    for c in range(L):
        row_str += "█ " if image[r, c] == 1 else "· "
    
    # Label the y-axis
    val_approx = ((r + 0.5) / H) * (2 * max_scale) - max_scale
    if r == H//2:
        print(f"{val_approx:+4.1f} | {row_str} <- ZERO MEAN")
    elif r == H-1:
        print(f"+3.5 | {row_str}")
    elif r == 0:
        print(f"-3.5 | {row_str}")
    else:
        print(f"     | {row_str}")
