import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Extended for xAI Prototype: Add torch-based NN for learning couplings, mock Grok API for scientific queries

# Define branches from Mind doc (16 branches)
branches = [
    "Cats", "Cars", "Work", "Education/Career", "AI/AR/Tech Projects",
    "Astronomy/Astrophotography", "Music/Creative", "Relationships",
    "Family/Genealogy", "Health/Mental Health", "Addiction/Recovery",
    "Finance/Workflows", "Magic: The Gathering", "Faith/Bible Study",
    "Utilities/How-to", "Retail/Oils Mapping"
]
N = len(branches)  # 16

# Parameters from equations
alpha, beta, gamma, delta, kappa = 1.0, 1.0, 0.5, 1.0, 0.5  # Attention coeffs
rho, eta = 0.9, 0.1  # Recursion update params

# Initial states (random for sim)
L = np.random.rand(N)  # Logic leaves
P = np.random.rand(N)  # Perception
E = np.random.rand(N)  # Experience
R = np.random.rand(N)  # Recursion
W_L = np.random.rand(N)  # Weights for leaves
W_P = np.random.rand(N)
W_E = np.random.rand(N)
W_R = np.random.rand(N)
b = np.random.rand(N)  # Biases
phi = lambda x: np.tanh(x)  # Activation (non-linear)

# Goals vector V(t) - dummy
V = np.random.rand(N)

# Constraints C(t) - e.g., energy level (starts high, decays)
C = 10.0
D_max = lambda c: c * 0.5  # Max depth from constraints

# Coupling matrix Gamma as Parameter for optimization
Gamma = torch.nn.Parameter(torch.rand((N, N)) * 0.1)  # Initial small random
Gamma.data[:, 9] = 0.1  # Health influences all
Gamma.data[2, 7] = 0.2  # Relationships -> Work
Gamma.data[:, 13] = -0.05  # Faith dampens errors
optimizer = torch.optim.Adam([Gamma], lr=0.01)  # Optimizer for learning

# Stage-Task integration: Pick Work branch (index 2) as example stage
# Assume tasks done -> close stage -> emit E=1.0
stage_closed = False
E_stage = 0.0

# Mock Grok API for xAI integration: Generate scientific query based on branch
def mock_grok_api(prompt):
    # In real prototype, call xAI API: https://api.x.ai/v1/chat/completions
    # Here, mock response for scientific discovery
    return f"Mock Grok response: Analyzed '{prompt}' for universe understanding."

# Simulation loop - extended with NN learning and Grok calls
T = 10  # Time steps
attention_history = np.zeros((T, N))
M_history = np.zeros(T)
epsilon_history = np.zeros(T)
grok_queries = []  # Collect prototype queries

for t in range(T):
    # Dummy inputs: urgency, novelty, reward, cost, depth
    urgency = np.random.rand(N) * 2
    novelty = np.random.rand(N)
    reward = np.random.rand(N) * 3 - 1
    cost = np.random.rand(N) * 1.5
    depth = np.random.rand(N) * 5 + 1  # Branch depth

    # Attention scores S_i (eq 4)
    align_scores = np.dot(V, np.random.rand(N))  # Dummy alignment
    S = alpha * align_scores + beta * urgency + gamma * novelty + delta * reward - kappa * cost / C

    # Softmax for a_i (eq 5)
    a = np.exp(S) / np.sum(np.exp(S))
    attention_history[t] = a

    # Leaf fusion y_i (eq 3)
    y = phi(W_L * L + W_P * P + W_E * E + W_R * R + b)

    # Cross-branch coupling (eq 10) - now with torch
    y_tensor = torch.tensor(y, dtype=torch.float32)
    y_tilde = y_tensor + torch.matmul(Gamma, y_tensor)  # Use learned Gamma

    # Global mind state M(t) (eq 1)
    M = np.sum(a * y_tilde.detach().numpy())
    M_history[t] = M

    # Prediction and error (eq 8) - dummy observed o*
    g = lambda m: m * 0.8  # Dummy predictor
    o_hat = g(M)
    o_star = np.random.normal(M, 0.5)  # Dummy observed (with noise)
    epsilon = o_star - o_hat
    epsilon_history[t] = epsilon

    # Updates (eq 6-7)
    events = np.random.rand(N) * epsilon  # Error-driven events
    E += events  # Experience update (oplus as + for sim)
    meta = np.abs(epsilon) * y_tilde.detach().numpy()  # Meta-analysis
    R = rho * R + eta * meta

    # Constraint check (eq 9) - prune if over
    total_depth = np.sum(a * depth)
    if total_depth > D_max(C):
        prune_idx = np.argmax(depth)  # Prune deepest (Thought Tree)
        y[prune_idx] *= 0.5  # Reduce output

    # Integrate Stage-Task: Check if Work branch (2) stage closes
    if not stage_closed and np.random.rand() > 0.7:  # Dummy condition
        stage_closed = True
        E_stage = 1.0  # Emit effect token

    # xAI Prototype Extension: Learn couplings with backprop on error
    loss = torch.tensor(epsilon ** 2, requires_grad=True)  # Minimize squared error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Generate scientific query if high attention on STEM branches (e.g., 3:Education,4:AI,5:Astronomy)
    stem_indices = [3,4,5]
    if np.max(a[stem_indices]) > 0.1:
        high_branch = branches[np.argmax(a[stem_indices]) + min(stem_indices)]
        prompt = f"Explore {high_branch} for scientific discovery: {M}"
        response = mock_grok_api(prompt)
        grok_queries.append(response)

    # Decay constraints (simulate fatigue)
    C *= 0.95

# Results
print("Final Mind State M(t):", M_history[-1])
print("Last Error (Epsilon):", epsilon_history[-1])
print("Stage Completion (Work branch E):", E_stage)
print("Learned Gamma (sample):", Gamma.data[0][:3].numpy())  # Sample learned couplings
print("Grok Queries (Prototype):", grok_queries)

# Plot attention history (described, not shown)
plt.figure()
for i in range(N):
    plt.plot(attention_history[:, i], label=branches[i])
plt.legend()
plt.title("Attention Weights Over Time - xAI Prototype")
plt.xlabel("Time Step")
plt.ylabel("Attention")
plt.show()
