"""Pure, vectorized implementation of Monte Carlo Tree Search for discrete actions.

This module implements a stateless, JIT-friendly MCTS that operates over batches of
states. It does not own any neural networks; instead, it accepts callables for
dynamics, value, and policy prediction.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812


def run_mcts(
    state: torch.Tensor,
    dynamics_fn: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    policy_fn: Callable[[torch.Tensor], torch.Tensor],
    num_simulations: int,
    num_actions: int,
    discount: float = 0.99,
    c_puct: float = 1.0,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs vectorized Monte Carlo Tree Search for discrete action spaces.

    This is a pure function. It builds a search tree in parallel for a batch of
    root states.

    Args:
        state: ``(B, feature_dim)`` root states to plan from.
        dynamics_fn: callable ``(state, action) -> (next_state, reward, done)``.
            Both inputs are batched. Returns batched next states, rewards, and done flags.
        value_fn: callable ``(state) -> value``. Returns ``(B,)`` or ``(B, 1)``.
        policy_fn: callable ``(state) -> logits``. Returns ``(B, num_actions)``
            prior logits.
        num_simulations: number of MCTS iterations to run.
        num_actions: size of the discrete action space.
        discount: reward discount factor (gamma).
        c_puct: exploration constant for PUCT formula.
        temperature: controls exploration in the final action selection.
            Higher values = more exploration. 0 = argmax.
        dirichlet_alpha: parameter for the Dirichlet noise distribution.
        dirichlet_fraction: weight of the Dirichlet noise added to the root prior.

    Returns:
        action: ``(B,)`` chosen actions.
        mcts_policy: ``(B, num_actions)`` visit counts / improved policy distribution.
        mcts_value: ``(B,)`` updated value estimate of the root node.
    """
    batch_size = state.shape[0]
    device = state.device

    # We flatten the tree into pre-allocated tensors.
    # A node is uniquely identified by (batch_idx, node_idx).
    # Since we expand exactly one node per simulation, the maximum number
    # of nodes per tree is `num_simulations + 1`.
    max_nodes = num_simulations + 1

    # Node-level statistics.
    # Shape: (B, max_nodes, num_actions)
    visit_counts = torch.zeros(
        (batch_size, max_nodes, num_actions), dtype=torch.float32, device=device
    )
    q_values = torch.zeros(
        (batch_size, max_nodes, num_actions), dtype=torch.float32, device=device
    )

    # Store the predicted priors and rewards for transitions we have explored.
    # Shape: (B, max_nodes, num_actions)
    priors = torch.zeros((batch_size, max_nodes, num_actions), dtype=torch.float32, device=device)
    rewards = torch.zeros((batch_size, max_nodes, num_actions), dtype=torch.float32, device=device)

    # Tree topology: transitions from (node, action) -> child_node
    # -1 means the child is not yet expanded.
    # Shape: (B, max_nodes, num_actions)
    children = torch.full(
        (batch_size, max_nodes, num_actions), -1, dtype=torch.long, device=device
    )

    # State storage: we keep the latent state for every expanded node.
    # Shape: (B, max_nodes, feature_dim)
    feature_dim = state.shape[1]
    node_states = torch.zeros(
        (batch_size, max_nodes, feature_dim), dtype=state.dtype, device=device
    )
    node_states[:, 0] = state

    # Expand the root node (index 0).
    with torch.no_grad():
        root_logits = policy_fn(state)
        root_priors = F.softmax(root_logits, dim=-1)

        # Add Dirichlet noise to the root node's prior if we are exploring (temp > 0)
        if temperature > 0.0 and dirichlet_fraction > 0.0:
            noise = torch.distributions.Dirichlet(
                torch.full((num_actions,), dirichlet_alpha, dtype=torch.float32, device=device)
            ).sample((batch_size,))
            root_priors = (1.0 - dirichlet_fraction) * root_priors + dirichlet_fraction * noise

        priors[:, 0] = root_priors

    # The next available node index to allocate.
    # Shape: (B,)
    next_node_idx = torch.ones((batch_size,), dtype=torch.long, device=device)

    batch_indices = torch.arange(batch_size, device=device)

    for _ in range(num_simulations):
        # 1. Selection: traverse from root to leaf
        curr_nodes = torch.zeros((batch_size,), dtype=torch.long, device=device)

        # We need to keep track of the path to backpropagate values.
        # Max path length is num_simulations.
        search_paths = torch.zeros((batch_size, num_simulations), dtype=torch.long, device=device)
        search_actions = torch.zeros(
            (batch_size, num_simulations), dtype=torch.long, device=device
        )
        depths = torch.zeros((batch_size,), dtype=torch.long, device=device)

        # Boolean mask tracking which environments have hit a leaf node in this simulation.
        # Once hitting a leaf, we stop selecting and wait for the expansion step.
        is_leaf = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        for _d in range(num_simulations):
            # If all paths have hit a leaf, we can stop the selection phase.
            if is_leaf.all():
                break

            # Only select actions for nodes that are NOT yet leaves.
            active_mask = ~is_leaf

            if not active_mask.any():
                break

            # Calculate PUCT for active nodes.
            # U(s, a) = c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
            active_nodes = curr_nodes[active_mask]
            active_batch = batch_indices[active_mask]

            n_s = visit_counts[active_batch, active_nodes].sum(dim=-1, keepdim=True)
            n_sa = visit_counts[active_batch, active_nodes]
            q_sa = q_values[active_batch, active_nodes]
            p_sa = priors[active_batch, active_nodes]

            # ── Min-Max Q-value Normalization ────────────────────────────────
            # To combine Q and U meaningfully when rewards can be large (e.g. > 100),
            # we normalize Q-values to [0, 1] using the min/max across the tree.
            b_q = q_values[active_batch]  # (active_B, max_nodes, num_actions)
            b_n = visit_counts[active_batch]

            # Mask out unvisited nodes to find true min/max of visited nodes
            visited_mask = b_n > 0

            q_for_min = torch.where(visited_mask, b_q, torch.inf)
            min_q = q_for_min.view(b_q.shape[0], -1).min(dim=-1)[0]

            q_for_max = torch.where(visited_mask, b_q, -torch.inf)
            max_q = q_for_max.view(b_q.shape[0], -1).max(dim=-1)[0]

            # Fallback if tree is empty (no visits yet)
            no_visits = min_q == torch.inf
            min_q[no_visits] = 0.0
            max_q[no_visits] = 0.0

            q_range = max_q - min_q
            q_range[q_range < 1e-8] = 1.0  # Prevent div-by-zero

            min_q = min_q.unsqueeze(-1)
            q_range = q_range.unsqueeze(-1)

            norm_q_sa = (q_sa - min_q) / q_range

            # For unvisited actions, set their normalized Q to 0.0 (pessimistic initialization)
            # This ensures they aren't pushed to massive negative values if min_q > 0.
            norm_q_sa = torch.where(n_sa == 0, torch.zeros_like(norm_q_sa), norm_q_sa)
            # ─────────────────────────────────────────────────────────────────

            u_sa = c_puct * p_sa * torch.sqrt(n_s) / (1.0 + n_sa)
            puct_scores = norm_q_sa + u_sa

            # Select best action
            best_actions = puct_scores.argmax(dim=-1)

            # Record the step in the path
            search_paths[active_batch, depths[active_mask]] = active_nodes
            search_actions[active_batch, depths[active_mask]] = best_actions

            # Traverse to child
            child_nodes = children[active_batch, active_nodes, best_actions]

            # If the child is -1, we've found an unexpanded leaf node!
            new_leaves = child_nodes == -1

            # For environments that just found a leaf, mark them so they stop searching.
            is_leaf[active_batch[new_leaves]] = True

            # For environments that DID NOT find a leaf, continue traversing.
            continue_mask = ~new_leaves
            if continue_mask.any():
                continue_batch = active_batch[continue_mask]
                curr_nodes[continue_batch] = child_nodes[continue_mask]
                depths[continue_batch] += 1

        # 2. Expansion and Evaluation
        # curr_nodes contains the node we expanded FROM.
        # best_actions contains the action we took FROM curr_nodes to hit the leaf.
        # We need to get the actual actions taken at the leaf edge.
        leaf_actions = search_actions[batch_indices, depths]
        leaf_parents = search_paths[batch_indices, depths]

        # Get the states of the parent nodes
        parent_states = node_states[batch_indices, leaf_parents]

        with torch.no_grad():
            # Step the dynamics model
            next_s, reward, done = dynamics_fn(parent_states, leaf_actions)
            if reward.ndim == 2 and reward.shape[1] == 1:
                reward = reward.squeeze(1)
            if done.ndim == 2 and done.shape[1] == 1:
                done = done.squeeze(1)

            # Evaluate the new state
            leaf_value = value_fn(next_s)
            if leaf_value.ndim == 2 and leaf_value.shape[1] == 1:
                leaf_value = leaf_value.squeeze(1)

            # Mask value if state is terminal
            leaf_value = leaf_value * (~done).float()

            leaf_logits = policy_fn(next_s)
            leaf_priors = F.softmax(leaf_logits, dim=-1)

        # Store the new node and its evaluations in the tree
        new_nodes = next_node_idx.clone()
        node_states[batch_indices, new_nodes] = next_s
        priors[batch_indices, new_nodes] = leaf_priors
        rewards[batch_indices, leaf_parents, leaf_actions] = reward
        children[batch_indices, leaf_parents, leaf_actions] = new_nodes

        # Increment node counters
        next_node_idx += 1

        # 3. Backpropagation
        # Walk back up the search paths and update Q values and visit counts.
        # Start from the leaf node's value and backpropagate it.
        returns = leaf_value.clone()

        # Iterate backwards through the path
        # Find the maximum depth any environment reached to know how far to loop back
        max_depth = int(depths.max().item())

        for d in range(max_depth, -1, -1):
            # Only update paths that actually reached this depth
            valid_depth_mask = depths >= d
            if not valid_depth_mask.any():
                continue

            valid_batch = batch_indices[valid_depth_mask]
            nodes_at_d = search_paths[valid_batch, d]
            actions_at_d = search_actions[valid_batch, d]

            # returns = r + gamma * returns
            r = rewards[valid_batch, nodes_at_d, actions_at_d]
            returns[valid_batch] = r + discount * returns[valid_batch]

            # Update Q-values using incremental mean
            old_n = visit_counts[valid_batch, nodes_at_d, actions_at_d]
            old_q = q_values[valid_batch, nodes_at_d, actions_at_d]

            new_n = old_n + 1.0
            new_q = old_q + (returns[valid_batch] - old_q) / new_n

            visit_counts[valid_batch, nodes_at_d, actions_at_d] = new_n
            q_values[valid_batch, nodes_at_d, actions_at_d] = new_q

    # Simulation done. Extract the improved policy from the root node.
    root_visit_counts = visit_counts[:, 0]  # (B, num_actions)

    # Calculate MCTS Policy (improved policy)
    if temperature == 0.0:
        # Argmax
        mcts_policy = torch.zeros_like(root_visit_counts)
        best_actions = root_visit_counts.argmax(dim=-1)
        mcts_policy.scatter_(1, best_actions.unsqueeze(-1), 1.0)
    else:
        # Softmax over counts^(1/temp)
        count_temp = root_visit_counts ** (1.0 / temperature)
        mcts_policy = count_temp / (count_temp.sum(dim=-1, keepdim=True) + 1e-8)

    # Sample final action
    # For training we usually sample (temp=1.0), for eval we take argmax (temp=0.0)
    if temperature == 0.0:
        final_actions = root_visit_counts.argmax(dim=-1)
    else:
        # torch.multinomial expects probabilities
        final_actions = torch.multinomial(mcts_policy, num_samples=1).squeeze(-1)

    # Calculate MCTS Value of the root node
    # Usually this is the value evaluated by the network at root, or the mean
    # Q-value weighted by the MCTS policy.
    # Let's use the mean Q-value weighted by the policy.
    root_q_values = q_values[:, 0]
    mcts_value = (mcts_policy * root_q_values).sum(dim=-1)

    return final_actions, mcts_policy, mcts_value
