import typing
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from vmas import make_env
from vmas.interactive_rendering import InteractiveEnv
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

from abm.agent import CustomDynamics, ForagingAgent, TargetAgent, add_process_noise_to_belief, compute_gradient, \
    compute_reward, observe, update_belief

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

STATE_COLOR_MAP = {
    0: (0, 0.6196078431372549, 0.45098039215686275),  # private #009E73
    1: (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),  # social #56B4E9
    2: (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),  # social #56B4E9
    3: (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),  # social #56B4E9
    4: (0.9019607843137255, 0.6235294117647059, 0),  # none #E69F00
}


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.y_dim = None
        self.x_dim = None
        self.initialization_box_ratio = 1
        self.viewer_zoom = None
        self.is_interactive = False

        self.action_size = 1
        self.n_agents = None
        self.min_dist_between_entities = None
        self.min_collision_distance = 0.005
        self.agent_radius = None
        self.max_speed = 0.05
        self.agent_collision_penalty = None

        self.n_targets = None
        self.target_speed = None
        self.targets_quality_type = None
        self.target_qualities = None

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.is_interactive = kwargs.pop("is_interactive", False)

        # world
        self.x_dim, self.y_dim = kwargs.pop("x_dim", 10), kwargs.pop("y_dim", 10)
        self.viewer_zoom = kwargs.pop("viewer_zoom", 1)
        self.viewer_size = kwargs.pop("viewer_size", (700, 700))
        self.visualize_semidims = kwargs.pop("visualize_semidims", True)
        self.min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.5)
        self.initialization_box_ratio = kwargs.pop("initialization_box_ratio", 1)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)

        # target
        self.n_targets = kwargs.pop("n_targets", 1)
        self.target_speed = kwargs.pop("target_speed", 1.0)

        # agents
        self.n_agents = kwargs.pop("n_agents", 5)
        self.max_speed = kwargs.pop("max_speed", 0.05)

        agent_kwargs = {
            "base_noise": kwargs.pop("base_noise", 0.1),

            "dist_noise_scale_priv": kwargs.pop("dist_noise_scale_priv", 2.0),
            "dist_noise_scale_soc": kwargs.pop("dist_noise_scale_soc", 2.0),

            "process_noise_scale": kwargs.pop("process_noise_scale", 0.02),
            "momentum": kwargs.pop("momentum", 0.9),
            "social_trans_scale": kwargs.pop("social_trans_scale", 1.0),
            "social_pos_scale": kwargs.pop("social_pos_scale", 5.0),
            "social_heading_scale": kwargs.pop("social_heading_scale", 5.0),
            "cost_priv": kwargs.pop("cost_priv", 1.0),
            "cost_belief": kwargs.pop("cost_belief", 0.5),
            "cost_heading": kwargs.pop("cost_heading", 0.25),
            "cost_pos": kwargs.pop("cost_pos", 0.1)
        }

        self.targets_quality_type = kwargs.pop("targets_quality", "HM")
        if self.targets_quality_type == "HM":
            self.target_qualities = [1.0 for _ in range(self.n_targets)]
        elif self.targets_quality_type == "HT":
            self.target_qualities = np.linspace(0.5, 1.5, self.n_targets).tolist()
        else:
            raise ValueError

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_dim,
            y_semidim=self.y_dim,
            collision_force=500,
            substeps=1,
            drag=0.25,
            dt=1,
            contact_margin=0.01,
        )

        for i in range(self.n_agents):
            agent = ForagingAgent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius * 2 if i == 0 and self.is_interactive else self.agent_radius),
                action_size=self.action_size,
                max_speed=self.max_speed,
                color=Color.BLUE,
                device=device,
                dynamics=CustomDynamics(),
                u_range=5,
                batch_dim=batch_dim,
                n_targets=self.n_targets,
                **agent_kwargs
            )
            world.add_agent(agent)

        for i in range(self.n_targets):
            target = TargetAgent(
                name=f"target_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius * self.target_qualities[i] * 4),
                color=Color.GRAY,
                alpha=0.1,
                render_action=True,
                max_speed=self.max_speed * self.target_speed,
                action_script=self.action_script_creator(),
                action_size=self.action_size,
                batch_dim=batch_dim,
                device=device,
                dynamics=CustomDynamics(),
                quality=self.target_qualities[i]
            )
            world.add_agent(target)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            entities=self.world.agents,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self.min_dist_between_entities,
            x_bounds=(-self.world.x_semidim * self.initialization_box_ratio,
                      self.world.x_semidim * self.initialization_box_ratio),
            y_bounds=(-self.world.y_semidim * self.initialization_box_ratio,
                      self.world.y_semidim * self.initialization_box_ratio),
        )

    def reward(self, agent: Agent):
        if not isinstance(agent, ForagingAgent):
            return torch.zeros_like(agent.state.pos[:, 0])

        # Avoid collisions with each other
        if self.agent_collision_penalty != 0:
            agent.collision_reward[:] = 0
            for a in self.world.agents:
                if a != agent:
                    agent.collision_reward[
                        self.world.get_distance(a, agent) < self.min_collision_distance
                        ] += self.agent_collision_penalty

        return agent.total_reward + agent.collision_reward


    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "target_distance_reward": agent.target_distance_reward,
            "target_reward": agent.target_reward,
            "channel_costs_reward": agent.channel_costs_reward,
            "collision_reward": agent.collision_reward
        }
        return info

    def observation(self, agent: Agent):
        if "target" in agent.name:
            return torch.zeros(1, 2, device=agent.device)
        return torch.zeros(agent.batch_dim, 2, device=agent.device)

    def process_action(self, agent: Agent):
        if self.is_interactive:
            probs = torch.zeros(5)
            probs[0] = 0.9 # Private
            probs[1] = 0.05 # Belief
            probs[2] = 0.0 # Heading
            probs[3] = 0.0 # Position
            probs[4] = 0.05 # None (no update)
            agent.action.u = torch.distributions.Categorical(probs=probs).sample((agent.batch_dim, 1))

        if "agent" in agent.name and isinstance(agent, ForagingAgent):
            # get observation based on the selected information channel (agent.action.u[:, 0])
            targets = [a for a in self.world.agents if "target" in a.name]
            other_agents = [a for a in self.world.agents if a != agent and "target" not in a.name]

            observe(agent, targets, other_agents)
            add_process_noise_to_belief(agent, self.target_speed)
            update_belief(agent)
            compute_gradient(agent)
            compute_reward(agent, targets)
        elif isinstance(agent, TargetAgent):
            agent.update_state_based_on_action(agent, self.world)


    def action_script_creator(self):
        action_dim = self.action_size
        def action_script(agent, world):
            agent.action.u = torch.zeros((agent.batch_dim, action_dim))
        return action_script

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering
        geoms: List[Geom] = []

        belief_visualized = False
        # add velocity vectors
        for agent in self.world.agents:
            if ("agent" in agent.name) and (not belief_visualized):
                self.visualize_belief(agent, env_index, geoms)
                belief_visualized = True

            self.draw_agent_velocity(agent, env_index, geoms, color=Color.BLACK, width=1)

            if isinstance(agent, ForagingAgent) and agent.action.u is not None:
                agent.color = STATE_COLOR_MAP[int(agent.action.u[env_index, 0].item())]

        return geoms

    def draw_agent_velocity(self, agent, env_index: int, geoms, color, width=1):
        from vmas.simulator import rendering
        norm_vel = agent.state.vel / torch.linalg.norm(agent.state.vel, dim=-1).unsqueeze(1)
        line = rendering.Line(
            agent.state.pos[env_index],
            agent.state.pos[env_index] + norm_vel[env_index] * self.agent_radius,
            width=width,
        )
        xform = rendering.Transform()
        line.add_attr(xform)
        line.set_color(*color.value)
        geoms.append(line)

    def visualize_belief(self, agent, env_index: int, geoms):
        from vmas.simulator import rendering
        # Visualize Belief State (Covariance Ellipses)
        # Iterate over all targets in the belief
        means = agent.belief_target_pos[env_index]  # (n_targets, 2)
        covs = agent.belief_target_covariance[env_index]  # (n_targets, 2, 2)

        for k in range(agent.n_targets):
            mean = means[k]
            cov = covs[k]

            # Eigen decomposition to find ellipse orientation and axes
            # We use torch.linalg.eigh for symmetric matrices
            try:
                eigvals, eigvecs = torch.linalg.eigh(cov)

                # Ensure eigenvalues are positive (numerical stability)
                eigvals = torch.clamp(eigvals, min=1e-6)

                # Scale factors: 2 std deviations (approx 95% confidence interval)
                # make_circle creates radius 1, so we scale by sqrt(eigval)*2
                std = 2  # 1
                scale_x = torch.sqrt(eigvals[0]) * std
                scale_y = torch.sqrt(eigvals[1]) * std

                # Rotation angle: arctan of the first eigenvector
                # Note: eigvecs columns are the eigenvectors
                angle = torch.atan2(eigvecs[1, 0], eigvecs[0, 0])

                # Create the ellipse
                ellipse = rendering.make_circle(radius=1.0, res=20)

                # Apply transforms
                xform = rendering.Transform()
                xform.set_scale(scale_x, scale_y)
                xform.set_rotation(angle)
                xform.set_translation(mean[0], mean[1])
                ellipse.add_attr(xform)

                # Set Color (Gray, semi-transparent)
                ellipse.set_color(*Color.BLUE.value, alpha=0.05)

                geoms.append(ellipse)
            except Exception:
                # Skip rendering this ellipse if math fails (e.g. singular matrix)
                pass


if __name__ == "__main__":
    scenario = Scenario()
    control_two_agents = False
    display_info = True
    save_render = True # True

    InteractiveEnv(
        make_env(
            scenario=scenario,
            num_envs=1,
            device="cpu",
            wrapper="gym",
            seed=0,
            wrapper_kwargs={"return_numpy": False},
            x_dim=5,
            y_dim=5,
            target_speed=0.1,
            n_agents=10,
            n_targets=3,
            targets_quality = 'HT',
            is_interactive=True,
            initialization_box_ratio=0.8,
            viewer_zoom=1.05,
            viewer_size = (600, 600),
            visualize_semidims=True,
            min_dist_between_entities=0.1,
            agent_radius=0.01,
            max_speed=0.05,
            dist_noise_scale_priv=1.0,
            dist_noise_scale_soc=1.0,
            social_trans_scale=1.0
        ),
        control_two_agents=control_two_agents,
        display_info=display_info,
        save_render=save_render,
        render_name=f"{scenario}_interactive" if isinstance(scenario, str) else "interactive_3",
    )