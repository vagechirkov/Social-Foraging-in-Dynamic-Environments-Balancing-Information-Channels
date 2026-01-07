from functools import partial
from typing import List

import torch
from vmas.simulator.core import Agent
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.utils import X, Y
from torch.distributions import MultivariateNormal


def invert_2x2_matrix(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the analytical inverse of a batch of 2x2 matrices.
    Input: (..., 2, 2)
    Output: (..., 2, 2)
    """
    # Extract components
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]

    # Determinant
    det = a * d - b * c

    # Safe Inverse of Determinant
    det_safe = torch.where(torch.abs(det) < eps, torch.sign(det) * eps, det)
    det_inv = 1.0 / det_safe

    # Construct Inverse Matrix
    # [[d, -b], [-c, a]] * (1/det)
    inv_matrix = torch.empty_like(matrix)
    inv_matrix[..., 0, 0] = d * det_inv
    inv_matrix[..., 0, 1] = -b * det_inv
    inv_matrix[..., 1, 0] = -c * det_inv
    inv_matrix[..., 1, 1] = a * det_inv

    return inv_matrix


class ForagingAgent(Agent):
    def __init__(
            self,
            batch_dim,
            device,
            n_targets,
            base_noise: float = 0.1,  # sigma_0: Base noise floor
            dist_noise_scale_priv: float = 2.0,  # beta_priv: Private signal attenuation
            dist_noise_scale_soc: float = 2.0,  # beta_soc: Social signal attenuation
            process_noise_scale: float = 0.02,  # sigma_proc: Prediction uncertainty
            # --- Social Channel Multipliers (Relative to sigma_0) ---
            social_trans_scale: float = 1.0,  # Scaling for Belief Transfer
            social_pos_scale: float = 5.0,  # Scaling for Positional Proxy
            social_heading_scale: float = 5.0,  # Scaling for Heading Heuristic
            belief_selectivity_threshold: float = 3.0, # Mahalanobis distance threshold, std squared
            # --- Physics ---
            momentum: float = 0.9,  # alpha: Gradient smoothing
            # --- Information usage costs ---
            cost_priv: float = 0.1,
            cost_belief: float = 0.5,
            cost_heading: float = 0.25,
            cost_pos: float = 0.1,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_targets = n_targets
        self.momentum = momentum

        # Sensing Parameters
        self.base_noise = base_noise
        self.dist_noise_scale_priv = dist_noise_scale_priv
        self.dist_noise_scale_soc = dist_noise_scale_soc
        self.process_noise_scale = process_noise_scale

        # Derived base noise levels for social channels (before distance scaling)
        # These correspond to sigma_trans, sigma_pos, sigma_heading
        self.base_sigma_trans = base_noise * social_trans_scale
        self.base_sigma_pos = base_noise * social_pos_scale
        self.base_sigma_heading = base_noise * social_heading_scale
        self.belief_selectivity_threshold = belief_selectivity_threshold

        # 3. Energy Costs Vector [Priv, Belief, Heading, Pos, None]
        # Used for reward calculation/energy expenditure tracking
        self.channel_costs = torch.tensor(
            [cost_priv, cost_belief, cost_heading, cost_pos, 0.0],
            device=device
        )

        # Reward tracking
        self.target_distance_reward = torch.zeros(batch_dim, device=device)  # only distance
        self.target_reward = torch.zeros(batch_dim, device=device)  # distance & quality
        self.channel_costs_reward = torch.zeros(batch_dim, device=device)  # costs only
        self.collision_reward = torch.zeros(batch_dim, device=device)
        self.total_reward = torch.zeros(batch_dim, device=device)

        # OBSERVATIONS (Current Step)
        # Position z_{i,k}: (batch, n_targets, 2)
        self.obs_target_pos = torch.zeros(batch_dim, n_targets, 2, device=device)
        # Covariance Sigma_{obs} / R_k: (batch, n_targets, 2, 2)
        self.obs_target_covariance = torch.zeros(batch_dim, n_targets, 2, 2, device=device)
        # Quality o_{q,k}: (batch, n_targets)
        self.obs_target_qual = torch.zeros(batch_dim, n_targets, device=device)
        # Quality Variance R_{q,k}: (batch, n_targets)
        self.obs_target_qual_var = torch.zeros(batch_dim, n_targets, device=device)
        # A boolean mask: True = Update this target, False = Skip (act as None)
        # Shape: (batch, n_targets)
        self.obs_validity_mask = torch.ones(batch_dim, device=device, dtype=torch.bool)

        # BELIEF STATE
        # Means mu_{i,k}: (batch, n_targets, 2)
        self.belief_target_pos = torch.zeros(batch_dim, n_targets, 2, device=device)
        # Covariances Sigma_{i,k}: (batch, n_targets, 2, 2)
        # Initialize with identity matrices to represent initial uncertainty
        self.belief_target_covariance = torch.eye(2, device=device).expand(batch_dim, n_targets, 2, 2)

        # Quality weights q_{i,k}: (batch, n_targets)
        self.belief_target_qual = torch.zeros(batch_dim, n_targets, device=device)
        # Quality Variance sigma^2_{q, i, k}: (batch, n_targets)
        # Initialize with high variance (uncertainty)
        self.belief_target_qual_var = torch.ones(batch_dim, n_targets, device=device)


class AgentObservations:
    """Namespace for stateless observation strategies."""

    @staticmethod
    def private(agent, targets, indices=None):
        """
        Implements Private Sensing (Z_priv).
        The agent scans the environment and receives noisy estimates for all resources.
        Noise scales with distance.
        """
        device = agent.device

        if indices is None:
            indices = torch.arange(agent.batch_dim, device=device)
        batch_dim = len(indices)
        agent_pos = agent.state.pos[indices]

        target_pos = torch.zeros(batch_dim, len(targets), 2, device=device)
        target_covariance = torch.zeros(batch_dim, len(targets), 2, 2, device=device)
        target_qual = torch.zeros(batch_dim, len(targets), device=device)
        target_qual_var = torch.zeros(batch_dim, len(targets), device=device)

        for i, t in enumerate(targets):
            # Slice target data if indices are provided
            t_pos = t.state.pos if indices is None else t.state.pos[indices]

            # 1. Calculate Euclidean distance to target d_{ik}
            # agent.state.pos: (batch, 2)
            # t.state.pos: (batch, 2)
            diff = agent_pos - t_pos
            dist = torch.norm(diff, dim=1)  # (batch,)

            # 2. Determine variances based on distance
            # sigma_priv^2(d_ik); sigma_qual^2(d_ik) - assuming similar scaling for quality
            sigma_priv = agent.base_noise + agent.dist_noise_scale_priv * dist
            var_priv = sigma_priv ** 2

            # 3. Create the covariance matrix Sigma_{obs} = sigma^2 * I
            # We construct a diagonal matrix for each batch element
            # Shape: (batch, 2, 2)
            cov_i = torch.zeros(batch_dim, 2, 2, device=device)
            cov_i[:, 0, 0] = var_priv
            cov_i[:, 1, 1] = var_priv
            target_covariance[:, i, :, :] = cov_i

            # 4. Sample target position z_{i,k} ~ N(x_k, Sigma_{obs})
            noise_pos = torch.randn(batch_dim, 2, device=device) * sigma_priv.unsqueeze(1)
            target_pos[:, i, :] = t_pos + noise_pos

            # 5. Sample target quality o_{q,k} ~ N(q_k, sigma_qual^2)
            true_quality = getattr(t, 'quality', torch.ones(batch_dim, device=device))
            # Assuming t.quality is (batch_dim,)
            t_qual = true_quality if indices is None else true_quality[indices]

            noise_qual = torch.randn(batch_dim, device=device) * sigma_priv
            target_qual[:, i] = t_qual + noise_qual
            target_qual_var[:, i] = var_priv

        return target_pos, target_covariance, target_qual, target_qual_var

    @staticmethod
    def others_belief(agent, other_agents, other_agent_mask, indices=None):
        """
        Observes the full belief state of a selected neighbor.
        Z_belief = { (mu_j + noise, q_j + noise) }
        """
        device = agent.device
        n_targets = agent.n_targets

        if indices is None:
            indices = torch.arange(agent.batch_dim, device=device)
        batch_dim = len(indices)

        # 1. Gather neighbor data specifically for the active indices
        # Shape: (batch, n_neighbors, ...)

        # Stack all neighbors' positions: (batch, n_neighbors, 2)
        all_neighbors_pos = torch.stack([a.state.pos[indices] for a in other_agents], dim=1)

        # Stack all neighbors' belief means: (batch, n_neighbors, n_targets, 2)
        all_neighbors_belief_pos = torch.stack([a.belief_target_pos[indices] for a in other_agents], dim=1)

        # Stack all neighbors' belief covs: (batch, n_neighbors, n_targets, 2, 2)
        all_neighbors_belief_cov = torch.stack([a.belief_target_covariance[indices] for a in other_agents], dim=1)

        # Stack all neighbors' quality estimates: (batch, n_neighbors, n_targets)
        all_neighbors_belief_qual = torch.stack([a.belief_target_qual[indices] for a in other_agents], dim=1)

        # 2. Select the specific neighbor j using other_agent_mask
        # mask is (batch,), need to slice it by indices first
        batch_mask = other_agent_mask[indices] # (batch_subset,)

        # Expand mask for gathering
        # For pos: (batch, 1, 2)
        idx_pos = batch_mask.view(batch_dim, 1, 1).expand(-1, 1, 2)
        neighbor_pos = torch.gather(all_neighbors_pos, 1, idx_pos).squeeze(1) # (batch, 2)

        # For belief pos: (batch, 1, n_targets, 2)
        idx_belief_pos = batch_mask.view(batch_dim, 1, 1, 1).expand(-1, 1, n_targets, 2)
        neighbor_belief_pos = torch.gather(all_neighbors_belief_pos, 1, idx_belief_pos).squeeze(1)

        # For belief cov: (batch, 1, n_targets, 2, 2)
        idx_belief_cov = batch_mask.view(batch_dim, 1, 1, 1, 1).expand(-1, 1, n_targets, 2, 2)
        neighbor_belief_cov = torch.gather(all_neighbors_belief_cov, 1, idx_belief_cov).squeeze(1)

        # For belief qual: (batch, 1, n_targets)
        idx_belief_qual = batch_mask.view(batch_dim, 1, 1).expand(-1, 1, n_targets)
        neighbor_belief_qual = torch.gather(all_neighbors_belief_qual, 1, idx_belief_qual).squeeze(1)

        # 3. Calculate Transmission Noise based on distance d_ij
        # agent.state.pos[indices]: (batch, 2)
        d_ij = torch.norm(agent.state.pos[indices] - neighbor_pos, dim=1) # (batch,)

        # Sigma_trans(d_ij)
        sigma_trans = agent.base_sigma_trans + agent.dist_noise_scale_soc * d_ij
        var_trans = sigma_trans ** 2 # (batch,)

        # SELECTIVITY TO SOCIAL INFO
        # A. Calculate Difference Vector (Delta)
        # shape: (batch, n_targets, 2)
        delta = agent.belief_target_pos[indices] - neighbor_belief_pos

        # B. Invert Neighbor Covariance (Analytical Inverse)
        neigh_cov_inv = invert_2x2_matrix(neighbor_belief_cov)

        # C. Mahalanobis Distance: delta^T @ Sigma^-1 @ delta
        # (batch, n_targets, 1, 2) @ (batch, n_targets, 2, 2) @ (batch, n_targets, 2, 1)

        # term1 = Sigma^-1 @ delta
        term1 = torch.matmul(neigh_cov_inv, delta.unsqueeze(-1))

        # dist_sq = delta^T @ term1
        dist_sq = torch.matmul(delta.unsqueeze(-2), term1).squeeze(-1).squeeze(-1)

        # D. The Gate
        # If Distance > Threshold, the info is "Surprising/Novel" (Useful).
        # If Distance is small, it's either Redundant (we agree) or Useless (neighbor is lost, i.e. low precision).
        is_novel = dist_sq > agent.belief_selectivity_threshold

        agent.obs_validity_mask[indices] = is_novel.any(dim=1)

        # 4. Construct Observations

        # Position Observation: z ~ N(mu_j, Sigma_obs)
        # Sigma_obs = Sigma_j + sigma_trans^2 * I

        # Create noise covariance matrix (batch, 1, 2, 2) expanded to targets
        noise_cov = torch.zeros(batch_dim, 2, 2, device=device)
        noise_cov[:, 0, 0] = var_trans
        noise_cov[:, 1, 1] = var_trans
        noise_cov = noise_cov.unsqueeze(1).expand(-1, n_targets, -1, -1) # (batch, n_targets, 2, 2)

        # R_k = Sigma_j + Noise_Cov
        target_covariance = neighbor_belief_cov + noise_cov

        # Sample position noise
        pos_noise = torch.randn(batch_dim, n_targets, 2, device=device) * sigma_trans.view(batch_dim, 1, 1)
        target_pos = neighbor_belief_pos + pos_noise

        # Quality Observation: o_q ~ N(q_j, sigma_trans^2)
        qual_noise = torch.randn(batch_dim, n_targets, device=device) * sigma_trans.unsqueeze(1)
        target_qual = neighbor_belief_qual + qual_noise

        # Quality Variance R_q
        target_qual_var = var_trans.unsqueeze(1).expand(-1, n_targets)

        return target_pos, target_covariance, target_qual, target_qual_var

    @staticmethod
    def others_heading(agent, other_agents, other_agent_mask, indices=None):
        """
        Observes neighbor's velocity to infer intent.
        Updates only the target best aligned with velocity.
        """
        device = agent.device
        n_targets = agent.n_targets

        if indices is None:
            indices = torch.arange(agent.batch_dim, device=device)
        batch_dim = len(indices)

        # 1. Gather neighbor pos and vel
        all_neighbors_pos = torch.stack([a.state.pos[indices] for a in other_agents], dim=1)
        all_neighbors_vel = torch.stack([a.state.vel[indices] for a in other_agents], dim=1)

        batch_mask = other_agent_mask[indices]

        idx_vec = batch_mask.view(batch_dim, 1, 1).expand(-1, 1, 2)
        neighbor_pos = torch.gather(all_neighbors_pos, 1, idx_vec).squeeze(1)  # (batch, 2)
        neighbor_vel = torch.gather(all_neighbors_vel, 1, idx_vec).squeeze(1)  # (batch, 2)

        # Normalize velocity
        vel_norm = torch.norm(neighbor_vel, dim=1, keepdim=True)
        vel_dir = neighbor_vel / torch.clamp(vel_norm, min=1e-6)  # (batch, 2)

        # 2. Find best aligned target k* in SELF belief
        # agent.belief_target_pos[indices]: (batch, n_targets, 2)
        self_belief_mu = agent.belief_target_pos[indices]

        # Vectors from neighbor to all my belief targets
        # (batch, n_targets, 2) - (batch, 1, 2) -> (batch, n_targets, 2)
        vec_to_targets = self_belief_mu - neighbor_pos.unsqueeze(1)

        # Project onto velocity direction
        # (batch, n_targets, 2) * (batch, 1, 2) -> (batch, n_targets) sum dim 2
        projections = (vec_to_targets * vel_dir.unsqueeze(1)).sum(dim=2)

        # Find index of max projection k*
        # (batch,)
        best_target_idx = torch.argmax(projections, dim=1)

        # 3. Construct Observations
        target_pos = torch.zeros(batch_dim, n_targets, 2, device=device)
        # Initialize covariance with infinity (effectively) to signify NO info for non-selected
        HUGE_VAR = 1e6
        target_covariance = torch.eye(2, device=device).view(1, 1, 2, 2).expand(batch_dim, n_targets, 2, 2) * HUGE_VAR

        target_qual = torch.zeros(batch_dim, n_targets, device=device)
        target_qual_var = torch.ones(batch_dim, n_targets, device=device) * HUGE_VAR

        # Process the selected target k*
        # We need to construct the specific observation for the winning index per batch

        # Heading noise parameters; no attenuation by distance here
        sigma_heading = agent.base_sigma_heading
        var_heading = sigma_heading ** 2

        # For the selected target, z = p_j + max(0, projection) * v_dir
        # We need to gather the specific projection value
        max_proj = torch.gather(projections, 1, best_target_idx.unsqueeze(1)).squeeze(1)
        max_proj = torch.clamp(max_proj, min=0)

        # Calculate calculated position z
        # (batch, 2) + (batch, 1) * (batch, 2)
        z_k_star = neighbor_pos + max_proj.unsqueeze(1) * vel_dir

        # Quality observation o_q ~ N(1, sigma^2)
        o_q_star = torch.normal(mean=1.0, std=sigma_heading, size=(batch_dim,), device=device)

        # 4. Scatter into the return tensors
        # We use scatter_ to put the values in the right k index

        # Position: scatter (batch, 1, 1) into (batch, n_targets, 2)
        target_pos.scatter_(1, best_target_idx.view(batch_dim, 1, 1).expand(-1, 1, 2), z_k_star.unsqueeze(1))

        # Covariance: Need to set the specific 2x2 block to identity * noise
        good_cov = torch.eye(2, device=device).unsqueeze(0).expand(batch_dim, -1, -1) * var_heading  # (batch, 2, 2)

        # Expand index for cov: (batch, 1, 2, 2)
        idx_cov = best_target_idx.view(batch_dim, 1, 1, 1).expand(-1, 1, 2, 2)
        target_covariance.scatter_(1, idx_cov, good_cov.unsqueeze(1))

        # Quality
        target_qual.scatter_(1, best_target_idx.unsqueeze(1), o_q_star.unsqueeze(1))
        target_qual_var.scatter_(1, best_target_idx.unsqueeze(1),
                                 torch.tensor(var_heading, device=device).expand(batch_dim, 1))

        return target_pos, target_covariance, target_qual, target_qual_var

    @staticmethod
    def others_location(agent, other_agents, other_agent_mask, indices=None):
        """
        Positional Proxy. Observes neighbor's static position.
        Updates only the spatially nearest resource in self belief.
        """
        device = agent.device
        n_targets = agent.n_targets

        if indices is None:
            indices = torch.arange(agent.batch_dim, device=device)
        batch_dim = len(indices)


        # 1. Gather neighbor pos
        all_neighbors_pos = torch.stack([a.state.pos[indices] for a in other_agents], dim=1)
        batch_mask = other_agent_mask[indices]
        idx_vec = batch_mask.view(batch_dim, 1, 1).expand(-1, 1, 2)
        neighbor_pos = torch.gather(all_neighbors_pos, 1, idx_vec).squeeze(1)  # (batch, 2)

        # 2. Find nearest target k* in SELF belief
        self_belief_mu = agent.belief_target_pos[indices]  # (batch, n_targets, 2)

        # Dist to all targets
        dists = torch.norm(self_belief_mu - neighbor_pos.unsqueeze(1), dim=2)  # (batch, n_targets)

        # Argmin
        best_target_idx = torch.argmin(dists, dim=1)  # (batch,)

        # 3. Construct Observations
        HUGE_VAR = 1e6
        target_pos = torch.zeros(batch_dim, n_targets, 2, device=device)
        target_covariance = torch.eye(2, device=device).view(1, 1, 2, 2).expand(batch_dim, n_targets, 2, 2) * HUGE_VAR
        target_qual = torch.zeros(batch_dim, n_targets, device=device)
        target_qual_var = torch.ones(batch_dim, n_targets, device=device) * HUGE_VAR

        # Position noise parameters; no attenuation by distance here
        sigma_pos = agent.base_sigma_pos
        var_pos = sigma_pos ** 2

        # Observation is just neighbor pos
        z_k_star = neighbor_pos
        o_q_star = torch.normal(mean=1.0, std=sigma_pos, size=(batch_dim,), device=device)

        # 4. Scatter
        # Position
        target_pos.scatter_(1, best_target_idx.view(batch_dim, 1, 1).expand(-1, 1, 2), z_k_star.unsqueeze(1))

        # Covariance
        good_cov = torch.eye(2, device=device).unsqueeze(0).expand(batch_dim, -1, -1) * var_pos
        idx_cov = best_target_idx.view(batch_dim, 1, 1, 1).expand(-1, 1, 2, 2)
        target_covariance.scatter_(1, idx_cov, good_cov.unsqueeze(1))

        # Quality
        target_qual.scatter_(1, best_target_idx.unsqueeze(1), o_q_star.unsqueeze(1))
        target_qual_var.scatter_(1, best_target_idx.unsqueeze(1),
                                 torch.tensor(var_pos, device=device).expand(batch_dim, 1))

        return target_pos, target_covariance, target_qual, target_qual_var


def observe(agent: ForagingAgent, targets: List[ForagingAgent], other_agents: List[Agent]):
    """
    Collects observations based on agent.action.u[:, 0].
    Channels: 0: Priv, 1: Belief, 2: Heading, 3: Pos, 4: None
    """
    # reset validity mask
    agent.obs_validity_mask[:] = True
    # get the most recent actions
    channel_indices = agent.action.u[:, 0].long()

    if len(other_agents) != 0:
        random_other_agent_ind = torch.randint(len(other_agents), (agent.batch_dim,), device=agent.device)
    else:
        random_other_agent_ind = torch.zeros(agent.batch_dim, device=agent.device).long()

    # Channel 4: None (observations remain the same; belief is not updated).
    channel_observation_functions = {
        0: partial(AgentObservations.private, targets=targets),
        1: partial(AgentObservations.others_belief, other_agents=other_agents,
                   other_agent_mask=random_other_agent_ind),
        2: partial(AgentObservations.others_heading, other_agents=other_agents,
                   other_agent_mask=random_other_agent_ind),
        3: partial(AgentObservations.others_location, other_agents=other_agents,
                   other_agent_mask=random_other_agent_ind)
    }

    for ch_idx, obs_func in channel_observation_functions.items():
        mask = (channel_indices == ch_idx)
        if mask.any():
            indices = torch.nonzero(mask).flatten()
            p, c, q, qv = obs_func(agent, indices=indices)

            agent.obs_target_pos[indices] = p
            agent.obs_target_covariance[indices] = c
            agent.obs_target_qual[indices] = q
            agent.obs_target_qual_var[indices] = qv


def add_process_noise_to_belief(agent: ForagingAgent, target_speed):
    # This prevents the covariance from collapsing to zero and allows tracking moving targets
    Q_scale = agent.process_noise_scale * (target_speed / 0.1)
    Q = torch.eye(2, device=agent.device).view(1, 1, 2, 2).expand_as(agent.belief_target_covariance) * (Q_scale ** 2)

    # Add process noise to current belief
    agent.belief_target_covariance = agent.belief_target_covariance + Q

    # Also add process noise to quality variance (assuming quality might drift slightly)
    # agent.belief_target_qual_var = agent.belief_target_qual_var + (Q_scale ** 2)

    # Add a tiny epsilon to the diagonal to prevent singularity
    epsilon = 1e-6
    I = torch.eye(2, device=agent.device).view(1, 1, 2, 2).expand_as(agent.belief_target_covariance)
    agent.belief_target_covariance = agent.belief_target_covariance + (I * epsilon)


def update_belief(agent: ForagingAgent):
    """
    Updates the internal GMM belief state using Kalman Filters for both
    spatial (vector) and quality (scalar) components.
    """
    # 1. Identify agents to update
    # Channels: 0: Private, 1: Belief, 2: Heading, 3: Pos, 4: None
    update_mask = (agent.action.u[:, 0].long() != 4) & agent.obs_validity_mask
    update_indices = torch.nonzero(update_mask).flatten()

    if update_indices.numel() == 0:
        return

    # TARGET POSITION UPDATE
    # Sigma_t: (batch, n_targets, 2, 2)
    sigma_t = agent.belief_target_covariance[update_indices]
    # R_k (Sigma_obs): (batch, n_targets, 2, 2)
    R_k = agent.obs_target_covariance[update_indices]
    # mu_t: (batch, n_targets, 2)
    mu_t = agent.belief_target_pos[update_indices]
    # z_k: (batch, n_targets, 2)
    z_k = agent.obs_target_pos[update_indices]

    # 1. Kalman Gain Calculation
    # Innovation Covariance S = Sigma_t + R_k
    S = sigma_t + R_k

    # Calculate Kalman Gain K = Sigma_t @ S^-1
    S_inv = invert_2x2_matrix(S)
    K = torch.matmul(sigma_t, S_inv)

    # 2. Update Mean
    # Innovation y = z_k - mu_t
    y = (z_k - mu_t).unsqueeze(-1)

    # correction = K * y
    correction = torch.matmul(K, y).squeeze(-1)

    agent.belief_target_pos[update_indices] = mu_t + correction

    # 3. Update Covariance
    # Sigma_{t+1} = (I - K) * Sigma_t
    I = torch.eye(2, device=agent.device).view(1, 1, 2, 2).expand_as(K)
    agent.belief_target_covariance[update_indices] = torch.matmul(I - K, sigma_t)

    # QUALITY UPDATE
    # sigma^2_{q, i, k} (batch, n_targets)
    sigma_q_t = agent.belief_target_qual_var[update_indices]
    # R_{q,k} (batch, n_targets)
    R_q_k = agent.obs_target_qual_var[update_indices]
    # q_t (batch, n_targets)
    q_t = agent.belief_target_qual[update_indices]
    # o_{q,k} (batch, n_targets)
    o_q = agent.obs_target_qual[update_indices]

    # 1. Kalman Gain Calculation
    # K_{q,k} = sigma^2_t / (sigma^2_t + R_{q,k})
    K_q = sigma_q_t / torch.clamp(sigma_q_t + R_q_k, 1e-8) # Avoid div by zero

    # 2. Update Mean
    # q_{t+1} = q_t + K_q * (o_q - q_t)
    agent.belief_target_qual[update_indices] = q_t + K_q * (o_q - q_t)

    # 3. Update Variance
    # sigma^2_{t+1} = (1 - K_q) * sigma^2_t
    agent.belief_target_qual_var[update_indices] = (1 - K_q) * sigma_q_t


def compute_gradient(agent: ForagingAgent):
    """
    Fully vectorized gradient computation.
    """
    batch_dim = agent.batch_dim
    n_targets = agent.n_targets
    device = agent.device

    # 1. Prepare Inputs for Broadcasting
    # Pos: (batch, 2) -> (batch, n_targets, 2)
    pos_expanded = agent.state.pos.unsqueeze(1).expand(-1, n_targets, -1)

    mu = agent.belief_target_pos          # (batch, n_targets, 2)
    sigma = agent.belief_target_covariance # (batch, n_targets, 2, 2)
    q = agent.belief_target_qual          # (batch, n_targets)

    # 2. Calculate Difference Vector
    diff = pos_expanded - mu # (batch, n_targets, 2)

    # 3. Analytical Inverse of Sigma (Vectorized over targets)
    sigma_inv = invert_2x2_matrix(sigma) # (batch, n_targets, 2, 2)

    # 4. Compute Gradient Component Vector: -Sigma^-1 @ diff
    # We treat diff as a column vector: (batch, n_targets, 2, 1)
    # Result: (batch, n_targets, 2, 1) -> squeeze to (batch, n_targets, 2)
    grad_vectors = -torch.matmul(sigma_inv, diff.unsqueeze(-1)).squeeze(-1)

    # 5. Compute Log Probability (Manually, Vectorized)
    # Formula: -0.5 * (log|Sigma| + (x-mu)^T Sigma^-1 (x-mu) + 2*log(2pi))

    # A. Determinant
    a = sigma[..., 0, 0]
    b = sigma[..., 0, 1]
    c = sigma[..., 1, 0]
    d = sigma[..., 1, 1]
    det = a * d - b * c
    det_safe = torch.where(torch.abs(det) < 1e-6, torch.sign(det) * 1e-6, det)
    log_det = torch.log(torch.abs(det_safe)) # (batch, n_targets)

    # B. Mahalanobis Distance
    # (x-mu)^T * Sigma^-1 * (x-mu)
    # Since grad_vectors = -Sigma^-1 * (x-mu),
    # Mahalanobis = (x-mu) * (-grad_vectors)
    # Dot product sum over dim 2
    mahalanobis = -(diff * grad_vectors).sum(dim=2) # (batch, n_targets)

    # C. Final Log Prob
    log_2pi = 1.837877
    log_prob = -0.5 * (log_det + mahalanobis + 2 * log_2pi)

    # 6. Weighting
    log_q = torch.log(torch.clamp(q, min=1e-10))
    log_weights = log_q + log_prob # (batch, n_targets)

    # 7. Log-Sum-Exp Trick (Vectorized)
    # Find max over targets dim
    max_log_weights, _ = torch.max(log_weights, dim=1, keepdim=True)

    # Exponentials
    stable_coeffs = torch.exp(log_weights - max_log_weights) # (batch, n_targets)

    # Weighted Sum of Gradient Vectors
    # (batch, n_targets, 1) * (batch, n_targets, 2) -> sum over n_targets
    total_grad_direction = (stable_coeffs.unsqueeze(-1) * grad_vectors).sum(dim=1)

    # 8. Normalize and Update
    grad_norm = torch.norm(total_grad_direction, dim=1, keepdim=True)
    grad_norm = torch.clamp(grad_norm, min=1e-6)
    direction = total_grad_direction / grad_norm

    agent.state.vel = agent.momentum * agent.state.vel + (1 - agent.momentum) * direction * agent.max_speed


def compute_reward(agent: ForagingAgent, targets):
    t_pos = torch.stack([t.state.pos for t in targets], dim=1)
    t_qual = torch.stack([t.quality for t in targets], dim=1)

    # Calculate Distances
    # agent.state.pos is (batch, 2) -> unsqueeze to (batch, 1, 2) for broadcasting
    diff = agent.state.pos.unsqueeze(1) - t_pos
    dist = torch.norm(diff, dim=2)  # (batch, n_targets)

    # Find Nearest Target
    # (batch, 1)
    nearest_target_id = torch.argmin(dist, dim=1, keepdim=True)

    # Gather data for the nearest target
    # (batch, 1) -> squeeze to (batch,)
    nearest_target_dist = torch.gather(dist, 1, nearest_target_id).squeeze(1)
    nearest_target_quality = torch.gather(t_qual, 1, nearest_target_id).squeeze(1)

    # Calculate Gathering Reward
    # Inverse square decay based on distance
    agent.target_distance_reward = (1.0 / (1.0 + nearest_target_dist ** 2.0))
    agent.target_reward = agent.target_distance_reward * nearest_target_quality  # Scale by quality

    # Calculate Information Channel Costs
    if agent.action.u is not None:
        # channel id: (batch,)
        channel_indices = agent.action.u[:, 0].long()
        # Retrieve cost for selected channel
        agent.channel_costs_reward = -agent.channel_costs[channel_indices]
    else:
        agent.channel_costs_reward = torch.zeros(agent.batch_dim, device=agent.device)

    agent.total_reward = agent.target_reward + agent.channel_costs_reward


class TargetAgent(Agent):
    def __init__(self, batch_dim, device, quality=1.0, persistence=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = torch.ones(batch_dim, device=device) * quality
        # converted to radians
        self.persistence_sigma = persistence * (torch.pi / 180.0)
        # Heading alpha for correlated random walk
        self.heading = torch.zeros(batch_dim, device=device)
        self.reset_heading(batch_dim, device)

    def reset_heading(self, batch_dim, device):
        # Initialize random heading [0, 2pi]
        self.heading = torch.rand(batch_dim, device=device) * 2 * torch.pi

    def update_state_based_on_action(self, t: Agent, world):
        """
        Correlated Random Walk (CRW)
        x(t+1) = x(t) + d(t)
        alpha(t) = alpha(t-1) + N(0, sigma^2)
        """
        # 1. Update Heading
        self.heading += torch.randn(t.batch_dim, device=t.device) * self.persistence_sigma

        # 2. Update Velocity Vector based on Heading
        # v_x = v * cos(alpha), v_y = v * sin(alpha)
        # speed is constant t.max_speed
        t.state.vel[:, X] = torch.cos(self.heading) * t.max_speed
        t.state.vel[:, Y] = torch.sin(self.heading) * t.max_speed

        # 3. Bounce off walls
        for dim, semidim in zip([X, Y], [world.x_semidim, world.y_semidim]):
            hit_mask = torch.abs(t.state.pos[:, dim]) > (semidim - t.shape.radius)
            if torch.any(hit_mask):
                t.state.vel[hit_mask, dim] *= -1
                self.heading[hit_mask] = torch.atan2(t.state.vel[hit_mask, Y], t.state.vel[hit_mask, X])


class CustomDynamics(Dynamics):
    def __init__(self, action_size: int = 1):
        super().__init__()
        self.action_size = action_size

    @property
    def needed_action_size(self) -> int:
        return self.action_size

    def process_action(self):
        #self.agent.state.force = self.agent.action.u[:, : self.needed_action_size]
        pass
