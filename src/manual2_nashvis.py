import numpy as np
import random
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Container data
containers = [
    {"mult": 10, "inhabitants": 1},
    {"mult": 80, "inhabitants": 6},
    {"mult": 37, "inhabitants": 3},
    {"mult": 17, "inhabitants": 1},
    {"mult": 90, "inhabitants": 10},
    {"mult": 31, "inhabitants": 2},
    {"mult": 50, "inhabitants": 4},
    {"mult": 20, "inhabitants": 2},
    {"mult": 73, "inhabitants": 4},
    {"mult": 89, "inhabitants": 8}
]

N_PLAYERS = 1000
N_CONTAINERS = len(containers)

# Initialize random strategies
strategies = [[random.randint(0, N_CONTAINERS - 1), random.randint(0, N_CONTAINERS - 1)] for _ in range(N_PLAYERS)]


import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Container data
containers = [
    {"mult": 10, "inhabitants": 1},
    {"mult": 80, "inhabitants": 6},
    {"mult": 37, "inhabitants": 3},
    {"mult": 17, "inhabitants": 1},
    {"mult": 90, "inhabitants": 10},
    {"mult": 31, "inhabitants": 2},
    {"mult": 50, "inhabitants": 4},
    {"mult": 20, "inhabitants": 2},
    {"mult": 73, "inhabitants": 4},
    {"mult": 89, "inhabitants": 8}
]

N_CONTAINERS = len(containers)
N_PLAYERS = 10000
N_ITERATIONS = 30
N_SAMPLES = 10

# Compute payoff with constraint: second container ≠ first
def compute_local_payoff_restricted(c1, c2, choice_counts, popularity, base_inhabitants):
    if c1 == c2:
        return -float("inf")
    total_opens = sum(choice_counts.values()) + 2
    profit = 0
    for j, c in enumerate((c1, c2)):
        mult = containers[c]["mult"]
        inhabitants = base_inhabitants[c] + choice_counts[c]
        divisor = inhabitants + (choice_counts[c] + 1) / total_opens * total_opens
        value = 10000 * mult / divisor
        if j == 1:
            value -= 50000
        profit += value
    return profit

# Reinitialize data
base_inhabitants = [c["inhabitants"] for c in containers]
strategies = [
    [random.randint(0, N_CONTAINERS - 1), None] for _ in range(N_PLAYERS)
]
for strat in strategies:
    first = strat[0]
    strat[1] = random.choice([x for x in range(N_CONTAINERS) if x != first])

choice_counts = Counter([c for pair in strategies for c in pair])

# Optimization loop
for _ in range(N_ITERATIONS):
    for i in range(N_PLAYERS):
        current = strategies[i]
        choice_counts[current[0]] -= 1
        choice_counts[current[1]] -= 1

        best_payoff = -float("inf")
        best_choice = current

        for _ in range(N_SAMPLES):
            c1 = random.randint(0, N_CONTAINERS - 1)
            valid_c2s = [c for c in range(N_CONTAINERS) if c != c1]
            c2 = random.choice(valid_c2s)
            profit = compute_local_payoff_restricted(c1, c2, choice_counts, choice_counts, base_inhabitants)
            if profit > best_payoff:
                best_payoff = profit
                best_choice = [c1, c2]

        strategies[i] = best_choice
        choice_counts[best_choice[0]] += 1
        choice_counts[best_choice[1]] += 1

# Analyze result
final_choice_counts = Counter(tuple(s) for s in strategies)
heatmap_matrix = np.zeros((N_CONTAINERS, N_CONTAINERS))
for (c1, c2), count in final_choice_counts.items():
    heatmap_matrix[c1][c2] = count

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_matrix, annot=True, fmt=".0f", cmap="YlGnBu", xticklabels=range(1, 11), yticklabels=range(1, 11))
plt.title("Heatmap of Container Choices (N = 1000, Restriction: Second ≠ First)")
plt.xlabel("Second Container Choice")
plt.ylabel("First Container Choice")
plt.tight_layout()
plt.show()
