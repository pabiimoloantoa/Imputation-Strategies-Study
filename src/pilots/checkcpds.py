from random_ground_truth_bn import build_random_ground_truth_bn, relabel_states, NODE_STATES
model = build_random_ground_truth_bn(node_states=NODE_STATES, max_indegree=3, edge_prob=0.4, seed=42)
model = relabel_states(model)
print(model.get_cpds("Income"))