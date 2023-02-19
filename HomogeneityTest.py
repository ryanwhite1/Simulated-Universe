# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:03:35 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


count = 800
radius = 450000
lowerbound = 5000
threshold = 30000
h0 = 4000

# ### -- HOMOGENEOUS -- ###
# closepopulation = int(np.sqrt(count))
# closedists = np.random.uniform((lowerbound / radius)**3, (threshold / radius)**3, closepopulation)
# fardists = np.random.uniform((lowerbound / radius)**3, 1, count - closepopulation)
# dists = np.append(closedists, fardists)
# # dists = np.random.uniform((lowerbound / radius)**3, 0.999, count)
# R = radius * np.cbrt(dists)

# rvs = np.random.uniform((threshold / radius)**3, 1, count)

# for i, dist in enumerate(R):
#     if (dist / radius)**3 > rvs[i]:
#         R[i] = radius * np.cbrt(np.random.uniform((threshold / radius)**3, (0.85)**3))
#         # if np.random.uniform(0, 1) > 0.9:
#         #     R[i] = radius * np.cbrt(np.random.uniform((lowerbound / radius)**3, (threshold / radius)**3))
        
### -- INHOMO -- ###
# proportion = 2 / np.sqrt(count)    # proportion of total galaxies that you want to be resolved, 1/sqrt(n) gives a good, scaleable number.
closepop = int(1 * np.sqrt(count)); farpop = int(count - closepop)  # find populations of each category
closescale = 5/5 * threshold    # the mean of the close distribution will actually be at about 3/5 of the threshold
# now, define the close distribution using scipy truncated exponential dist. b is a form of the upper bound.
# loc is the lower bound of the distribution and scale is the mean value after the lowerbound (i think?)
closedistribution = stats.truncexpon(b=(threshold - lowerbound)/closescale, loc=lowerbound, scale=closescale)
# now, to get the increasing shape we minus the random variables from the upper bound, and add the lower bound again to account for the shift
closedists = threshold - closedistribution.rvs(closepop) + lowerbound       # make 'closepop' number of random variables
# most of the steps below are analogous, but for the distant galaxy clusters
farscale = radius / 2
fardistribution = stats.truncexpon(b=(radius - threshold)/farscale, loc=threshold, scale=farscale)
fardists = fardistribution.rvs(farpop)
R = np.append(closedists, fardists)

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