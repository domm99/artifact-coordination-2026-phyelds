import math
import torch
from vmas import make_env
from phyelds.calculus import aggregate, neighbors
from phyelds.libraries.device import store, sense, local_id
from VMASRenderMonitor import VMASRenderMonitor
from phyelds.simulator import Simulator, VmasEnvironment
from phyelds.simulator.effects import RenderConfig, RenderMode
from phyelds.simulator.runner import vmas_runner, schedule_program_for_all
from phyelds.simulator.neighborhood import radius_neighborhood

def perturbation():
    angle = (torch.rand(1) * 2 - 1)* math.pi # ~ U(-pi,pi)
    return angle

def mean_velocity(velocities):
    t = torch.stack(velocities)
    return torch.mean(t, dim=0).squeeze()

def velocity_to_angle(vel, avg_vel):
    eps = 1e-6
    if torch.norm(avg_vel) > eps:
        direction = avg_vel / torch.norm(avg_vel)
    else:
        vel_norm = torch.norm(vel)

        if vel_norm > eps:
            direction = vel / vel_norm
        else:
            theta_rand = torch.rand(1, device=vel.device) * 2 * math.pi
            direction = torch.stack([
                torch.cos(theta_rand),
                torch.sin(theta_rand)
            ])
    return torch.atan2(direction[1], direction[0])

@aggregate
def action():
    myself = sense('agent')
    vel = myself.state.vel.squeeze()
    neighbors_info = neighbors(vel).exclude_self()
    if neighbors_info:
        velocities = [vel for vel in neighbors_info.values()]
        avg_vel = mean_velocity(velocities)
        theta = velocity_to_angle(vel, avg_vel)
        noise = perturbation()
        theta = theta + 0.1 * noise
        next_vel = [torch.cos(theta).item(), torch.sin(theta).item()]
        store("action", next_vel)
    else:
        vel = vel.flatten().tolist()
        store("action", vel)


env = make_env(
    scenario="flocking",
    continuos_actions=False,
    num_envs=1,
    n_agents=20,
    n_obstacles=0
    )

vmas_environment = VmasEnvironment(env)
sim = Simulator(vmas_environment)
sim.environment.set_neighborhood_function(radius_neighborhood(1.0))

# take agents positions
positions = [agent.state.pos for agent in vmas_environment.vmas_environment.agents]

schedule_program_for_all(sim, 0.0, 1.0, action)
sim.schedule_event(
    0.2, vmas_runner, sim, 1.0
)

VMASRenderMonitor(
    sim,
    RenderConfig(
        effects=[],
        mode=RenderMode.SAVE,
        save_as="vicsek",
        dt=0.1
    )
)

sim.run(200)