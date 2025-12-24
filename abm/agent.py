import torch
from vmas.simulator.core import Agent
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.utils import X, Y
from torch.distributions import MultivariateNormal


class ForagingAgent(Agent):
    def __init__(
            self,
            batch_dim,
            device,
            n_targets,
            base_noise=0.05,  # Base sensor noise
            dist_noise_scale=0.1,  # Variance increases by this factor per unit distance
            process_noise_scale=0.02,  # Uncertainty added per step (prediction step)
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_targets = n_targets

        # Hyperparameters
        self.base_noise = base_noise
        self.dist_noise_scale = dist_noise_scale
        self.process_noise_scale = process_noise_scale

        # Reward tracking
        self.target_reward = torch.zeros(batch_dim, device=device)
        self.collision_reward = torch.zeros(batch_dim, device=device)

        # OBSERVATIONS (Current Step)
        # Position z_{i,k}: (batch, n_targets, 2)
        self.obs_target_pos = torch.zeros(batch_dim, n_targets, 2, device=device)
        # Covariance Sigma_{obs} / R_k: (batch, n_targets, 2, 2)
        self.obs_target_covariance = torch.zeros(batch_dim, n_targets, 2, 2, device=device)
        # Quality o_{q,k}: (batch, n_targets)
        self.obs_target_qual = torch.zeros(batch_dim, n_targets, device=device)
        # Quality Variance R_{q,k}: (batch, n_targets)
        self.obs_target_qual_var = torch.zeros(batch_dim, n_targets, device=device)

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
    def private(agent, targets):
        """
        Implements Private Sensing (Z_priv).
        The agent scans the environment and receives noisy estimates for all resources.
        Noise scales with distance.
        """
        batch_dim = agent.batch_dim
        device = agent.device

        target_pos = torch.zeros_like(agent.belief_target_pos)
        target_covariance = torch.zeros_like(agent.belief_target_covariance)
        target_qual = torch.zeros_like(agent.belief_target_qual)
        target_qual_var = torch.zeros_like(agent.belief_target_qual_var)

        for i, t in enumerate(targets):
            # 1. Calculate Euclidean distance to target d_{ik}
            # agent.state.pos: (batch, 2)
            # t.state.pos: (batch, 2)
            diff = agent.state.pos - t.state.pos
            dist = torch.norm(diff, dim=1)  # (batch,)

            # 2. Determine variances based on distance
            # sigma_priv^2(d_ik)
            sigma_priv = agent.base_noise + agent.dist_noise_scale * dist
            var_priv = sigma_priv ** 2

            # sigma_qual^2(d_ik) - assuming similar scaling for quality
            sigma_qual = agent.base_noise + agent.dist_noise_scale * dist
            var_qual = sigma_qual ** 2


            # 3. Create the covariance matrix Sigma_{obs} = sigma^2 * I
            # We construct a diagonal matrix for each batch element
            # Shape: (batch, 2, 2)
            cov_i = torch.zeros(batch_dim, 2, 2, device=device)
            cov_i[:, 0, 0] = var_priv
            cov_i[:, 1, 1] = var_priv
            target_covariance[:, i, :, :] = cov_i

            # 4. Sample target position z_{i,k} ~ N(x_k, Sigma_{obs})
            noise_pos = torch.randn(batch_dim, 2, device=device) * sigma_priv.unsqueeze(1)
            target_pos[:, i, :] = t.state.pos + noise_pos

            # 5. Sample target quality o_{q,k} ~ N(q_k, sigma_qual^2)
            true_quality = getattr(t, 'quality', torch.ones(batch_dim, device=device))
            noise_qual = torch.randn(batch_dim, device=device) * sigma_qual
            target_qual[:, i] = true_quality + noise_qual
            target_qual_var[:, i] = var_qual

        return target_pos, target_covariance, target_qual, target_qual_var

    @staticmethod
    def others_belief(agent, other_agents, other_agent_mask):
        others_belief = torch.zeros_like(agent.belief_target_pos)

        for i, other_agent in enumerate(other_agents):
            others_belief = torch.where(
                other_agent_mask == i,
                other_agent.belief_target_pos,
                others_belief
            )

        return others_belief
    ...

    @staticmethod
    def others_heading(agent, other_agents, other_agent_mask):
        ...
    ...

    @staticmethod
    def others_location(agent, other_agents, other_agent_mask):
        ...
    ...


def observe(agent: ForagingAgent, world):
    """
    Collects observations for the agent.
    """
    # n_agents,
    # random_other_agent_ind = torch.randint(n_agents, (agent.batch_dim,))
    # other_agents = [a for a in world.agents if a != agent and "target" not in a.name]
    #
    # candidates_pos = torch.stack([
    #     AgentObservations.private(agent, targets),
    #     AgentObservations.others_belief(agent, other_agents, random_other_agent_ind),
    #     AgentObservations.others_heading(agent, other_agents, random_other_agent_ind),
    #     AgentObservations.others_location(agent, other_agents, random_other_agent_ind),
    # ], dim=1)

    # selected observation for each agent in the batch
    # agent.action.u[:, 0]

    targets = [a for a in world.agents if "target" in a.name]
    pos, covariance, quality, quality_var = AgentObservations.private(agent, targets)

    agent.obs_target_pos = pos
    agent.obs_target_covariance = covariance
    agent.obs_target_qual = quality
    agent.obs_target_qual_var = quality_var


def update_belief(agent: ForagingAgent, target_speed):
    """
    Updates the internal GMM belief state using Kalman Filters for both
    spatial (vector) and quality (scalar) components.
    """
    # PREDICTION STEP (Add Noise)
    # This prevents the covariance from collapsing to zero and allows tracking moving targets
    Q_scale = agent.process_noise_scale * (target_speed / 0.1)
    Q = torch.eye(2, device=agent.device).view(1, 1, 2, 2).expand_as(agent.belief_target_covariance) * (Q_scale ** 2)

    # Add process noise to current belief
    agent.belief_target_covariance = agent.belief_target_covariance + Q

    # Also add process noise to quality variance (assuming quality might drift slightly)
    # agent.belief_target_qual_var = agent.belief_target_qual_var + (Q_scale ** 2)

    # SPATIAL UPDATE
    # Sigma_t: (batch, n_targets, 2, 2)
    sigma_t = agent.belief_target_covariance
    # R_k (Sigma_obs): (batch, n_targets, 2, 2)
    R_k = agent.obs_target_covariance
    # mu_t: (batch, n_targets, 2)
    mu_t = agent.belief_target_pos
    # z_k: (batch, n_targets, 2)
    z_k = agent.obs_target_pos

    # 1. Kalman Gain Calculation
    # Innovation Covariance S = Sigma_t + R_k
    S = sigma_t + R_k

    # K = Sigma_t * S^-1
    try:
        S_inv = torch.linalg.inv(S)
        K = torch.matmul(sigma_t, S_inv)
    except RuntimeError:
        # Fallback for singular matrices
        S_inv = torch.linalg.pinv(S)
        K = torch.matmul(sigma_t, S_inv)

    # 2. Update Mean
    # Innovation y = z_k - mu_t
    y = (z_k - mu_t).unsqueeze(-1)

    # correction = K * y
    correction = torch.matmul(K, y).squeeze(-1)

    agent.belief_target_pos = mu_t + correction

    # 3. Update Covariance
    # Sigma_{t+1} = (I - K) * Sigma_t
    I = torch.eye(2, device=agent.device).view(1, 1, 2, 2).expand_as(K)
    agent.belief_target_covariance = torch.matmul(I - K, sigma_t)

    # QUALITY UPDATE
    # sigma^2_{q, i, k} (batch, n_targets)
    sigma_q_t = agent.belief_target_qual_var
    # R_{q,k} (batch, n_targets)
    R_q_k = agent.obs_target_qual_var
    # q_t (batch, n_targets)
    q_t = agent.belief_target_qual
    # o_{q,k} (batch, n_targets)
    o_q = agent.obs_target_qual

    # 1. Kalman Gain Calculation
    # K_{q,k} = sigma^2_t / (sigma^2_t + R_{q,k})
    K_q = sigma_q_t / torch.clamp(sigma_q_t + R_q_k, 1e-8) # Avoid div by zero

    # 2. Update Mean
    # q_{t+1} = q_t + K_q * (o_q - q_t)
    agent.belief_target_qual = q_t + K_q * (o_q - q_t)

    # 3. Update Variance
    # sigma^2_{t+1} = (1 - K_q) * sigma^2_t
    agent.belief_target_qual_var = (1 - K_q) * sigma_q_t


def compute_gradient(agent: ForagingAgent):
    """
    Computes the gradient of the utility surface (GMM) and updates action.

    Action is gradient ascent on:
    B(x) = sum_k q_k * N(x | mu_k, Sigma_k)

    Gradient:
    grad B(x) = sum_k q_k * grad_x N(x | ...)
    grad_x N(x) = - Sigma^{-1} (x - mu) * N(x)
    """
    batch_dim = agent.batch_dim
    n_targets = agent.n_targets

    # Agent position: (batch, 2)
    pos = agent.state.pos
    total_grad = torch.zeros(batch_dim, 2, device=agent.device)

    # Loop over GMM components (targets)
    for k in range(n_targets):
        # Extract component parameters
        mu = agent.belief_target_pos[:, k, :]       # (batch, 2)
        sigma = agent.belief_target_covariance[:, k, :, :] # (batch, 2, 2)
        q = agent.belief_target_qual[:, k]          # (batch,)

        # Create Multivariate Normal distribution
        # We need this to calculate the PDF value N(x | mu, Sigma)
        try:
            dist = MultivariateNormal(loc=mu, covariance_matrix=sigma)

            # Calculate PDF value: N(x)
            # log_prob returns (batch,), exp to get prob
            pdf_val = torch.exp(dist.log_prob(pos)) # (batch,)

            # Calculate gradient term: - Sigma^{-1} (x - mu)
            # (x - mu): (batch, 2)
            diff = pos - mu

            # Sigma inverse: (batch, 2, 2)
            sigma_inv = torch.linalg.inv(sigma)

            # term: (batch, 2, 2) @ (batch, 2, 1) -> (batch, 2, 1)
            term = torch.bmm(sigma_inv, diff.unsqueeze(-1)).squeeze(-1)
            term = -1 * term

            # Component gradient: q * term * pdf_val
            # (batch, 1) * (batch, 2) * (batch, 1) -> (batch, 2)
            component_grad = q.unsqueeze(1) * term * pdf_val.unsqueeze(1)

            total_grad += component_grad

        except ValueError as e:
            # Handle numerical instability
            pass

    # Normalize gradient to get direction
    grad_norm = torch.norm(total_grad, dim=1, keepdim=True)

    # Avoid division by zero
    grad_norm = torch.clamp(grad_norm, min=1e-6)
    direction = total_grad / grad_norm

    # Set agent's velocity
    # v_i(t) = v_max * (grad / ||grad||)
    alpha = 0.8
    # v_new = alpha * v_old + (1 - alpha) * v_target
    agent.state.vel = alpha * agent.state.vel + (1 - alpha) * direction * agent.max_speed


def compute_reward(agent: ForagingAgent, world):
    # TODO: compute reward as an inverse distance to closest target
    agent.target_reward = ...


class TargetAgent(Agent):
    def __init__(self, batch_dim, device, quality=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step_remaining_time = torch.zeros(batch_dim, device=device)
        self.quality = torch.ones(batch_dim, device=device) * quality

    def update_state_based_on_action(self, t: Agent, world):
        rotation_angle = torch.atan2(t.state.vel[:, Y], t.state.vel[:, X])
        rotation_angle = self.levy_walk(t, rotation_angle, dt=world.dt)

        update_velocity(t, rotation_angle, x_semidim=world.x_semidim - t.shape.radius)

        # normalize velocity
        t.state.vel *= t.max_speed

        # normalize velocity
        t.state.vel /= torch.linalg.norm(t.state.vel, dim=-1).unsqueeze(1)

    def levy_walk(self, t: Agent, rotation_angle, dt):
        """ Lévy flight behavior for the target."""
        rotation_angle = torch.where(
            t.current_step_remaining_time <= 0,
            torch.rand_like(rotation_angle) * 2 * torch.pi - torch.pi,
            rotation_angle,
            )

        t.current_step_remaining_time = torch.where(
            t.current_step_remaining_time <= 0,
            sample_pareto_limited(1, t.batch_dim, max_value=100, device=t.device) * dt,
            t.current_step_remaining_time - dt,
            )
        return rotation_angle


def update_velocity(_agent, rotation_angle, x_semidim):
    # update velocity based on the new angle
    _agent.state.vel[:, X] = torch.cos(rotation_angle)
    _agent.state.vel[:, Y] = torch.sin(rotation_angle)

    # bounce off walls
    if torch.any(torch.abs(_agent.state.pos) > x_semidim):
        _agent.state.vel = -_agent.state.pos


def sample_pareto_limited(a, batch_size, device, max_value=100):
    """
    Generate a single sample from a Pareto distribution, shift it by 1,
    and limit it to a maximum value.

    Parameters:
    - a: Shape parameter of the Pareto distribution.
    - max_value: Maximum value to clip the result (default=100).

    Returns:
    - A single float value following the described behavior.
    """
    # Sample from the Pareto distribution using the inverse CDF method
    u = torch.rand(batch_size, device=device)  # Uniform random number in [0, 1)
    pareto_sample = (1 / (1 - u)) ** (1 / a) - 1  # Pareto sample

    # Shift by 1 and apply the max limit
    return torch.min(pareto_sample + 1, torch.tensor(max_value))


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
