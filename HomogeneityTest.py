# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:03:35 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt


count = 800
radius = 450000
lowerbound = 5000
threshold = 30000
h0 = 4000


closepopulation = int(np.sqrt(count))
closedists = np.random.uniform((lowerbound / radius)**3, (threshold / radius)**3, closepopulation)
fardists = np.random.uniform((lowerbound / radius)**3, 1, count - closepopulation)
dists = np.append(closedists, fardists)
# dists = np.random.uniform((lowerbound / radius)**3, 0.999, count)
R = radius * np.cbrt(dists)

rvs = np.random.uniform((threshold / radius)**3, 1, count)

for i, dist in enumerate(R):
    if (dist / radius)**3 > rvs[i]:
        R[i] = radius * np.cbrt(np.random.uniform((threshold / radius)**3, (0.85)**3))
        # if np.random.uniform(0, 1) > 0.9:
        #     R[i] = radius * np.cbrt(np.random.uniform((lowerbound / radius)**3, (threshold / radius)**3))

tally = 0
total = 0
for i, dist in enumerate(R):
    if dist < threshold:
        total += int(np.random.exponential(8))
        tally += 1
        
print(tally, total)

fig, ax = plt.subplots()
# ax.scatter(R, h0 * R / 1e6, s=0.5)
ax.hist(R / 1e3, bins=40)
ax.set_xlim(0, 450)

ax.set_xlabel("Distance (kpc)")
ax.set_ylabel("Cluster count in bin")