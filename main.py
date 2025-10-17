# 8-ary 3D Signal Simulation (Simple Beginner Version)
# ----------------------------------------------------
# This code simulates transmission of 8 symbols placed
# on cube corners in 3D, adds Gaussian noise, detects
# received symbols, and plots results.

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# --------------------------
# Step 1: Basic Parameters
# --------------------------
A = 0.01           # Half cube side
M = 8              # Total number of points
N = 10000          # Total symbols
N0 = 2e-4          # Noise power

noise_var = N0 / 2
sigma = np.sqrt(noise_var)

print("Step 1: Parameters")
print("Amplitude A =", A)
print("Number of symbols =", M)
print("Total samples =", N)
print("Noise variance =", noise_var)
print("Sigma =", sigma)
print()

# --------------------------
# Step 2: Constellation Points
# --------------------------
# Cube has all ±A combinations in 3D
points = list(product([-A, A], repeat=3))
const_points = np.array(points)

print("Step 2: Constellation Points")
print(const_points)
print()

# --------------------------
# Step 3: Transmit Random Symbols
# --------------------------
tx_index = np.random.randint(0, M, N)
tx_data = const_points[tx_index]

print("Step 3: First few transmitted points:")
print(tx_data[:5])
print()

# --------------------------
# Step 4: Add Noise (AWGN)
# --------------------------
noise = np.random.normal(0, sigma, tx_data.shape)
rx_data = tx_data + noise

print("Step 4: First few received points:")
print(rx_data[:5])
print()

# --------------------------
# Step 5: ML Detection (Nearest point)
# --------------------------
def detect(received, const):
    detected = []
    for r in received:
        dist = np.linalg.norm(const - r, axis=1)
        idx = np.argmin(dist)
        detected.append(idx)
    return np.array(detected)

rx_index = detect(rx_data, const_points)

print("Step 5: First few detected indices:")
print(rx_index[:5])
print()

# --------------------------
# Step 6: Error Rate
# --------------------------
errors = np.sum(tx_index != rx_index)
Pe = errors / N

print("Step 6: Error Calculation")
print("Symbol errors:", errors)
print("Symbol error probability:", Pe)
print()

# --------------------------
# Step 7: Plot Noise Histogram
# --------------------------
plt.hist(noise.flatten(), bins=50, density=True, color='skyblue')
plt.title("Noise Distribution")
plt.xlabel("Noise value")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# --------------------------
# Step 8: Plot TX vs RX (2D)
# --------------------------
plt.scatter(tx_data[:, 0], tx_data[:, 1], c='red', label='Ideal', s=40)
plt.scatter(rx_data[:, 0], rx_data[:, 1], c='blue', alpha=0.3, label='Received', s=10)
plt.title("X-Y Plane View")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# Step 9: 3D Plot
# --------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(const_points[:, 0], const_points[:, 1], const_points[:, 2],
           c='red', s=80, label='Constellation')
ax.scatter(rx_data[:2000, 0], rx_data[:2000, 1], rx_data[:2000, 2],
           c='blue', alpha=0.3, s=5, label='Received')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Cube Transmission")
ax.legend()
plt.show()

# --------------------------
# Step 10: Decision Boundaries (2D Slice)
# --------------------------
# Plane boundaries at x=0, y=0, z=0
grid = np.linspace(-2*A, 2*A, 200)
X, Y = np.meshgrid(grid, grid)
Z = np.zeros_like(X)

all_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
distances = np.linalg.norm(all_points[:, None, :] - const_points[None, :, :], axis=2)
closest = np.argmin(distances, axis=1).reshape(X.shape)

plt.contourf(X, Y, closest, levels=8, cmap='tab10', alpha=0.4)
plt.scatter(const_points[:, 0], const_points[:, 1], c='red', s=60)
plt.title("Decision Boundaries (z = 0 slice)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

print("Step 10: Done. Boundaries are at x=0, y=0, z=0.\n")
