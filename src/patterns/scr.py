import random
from phyelds.calculus import aggregate
from phyelds.simulator import Simulator
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.spreading import broadcast
from phyelds.libraries.spreading import distance_to
from CustomRenderMonitor import CustomRenderMonitor
from phyelds.simulator.deployments import deformed_lattice
from phyelds.libraries.distances import neighbors_distances
from CustomDrawings import CustomDrawNodes, CustomDrawEdges
from phyelds.libraries.leader_election import elect_leaders
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.effects import RenderConfig, RenderMode
from phyelds.simulator.neighborhood import radius_neighborhood


random.seed(42)


@aggregate
def main():
    distances = neighbors_distances()
    leader = elect_leaders(4, distances)
    potential = distance_to(leader, distances)
    nodes = count_nodes(potential)
    area_value = broadcast(leader, nodes, distances)
    return area_value

simulator = Simulator()
# deformed lattice
simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
deformed_lattice(simulator, 20, 20, 1, 0.01)
# put source
for node in simulator.environment.nodes.values():
    node.data = {"source": False, "target": False}
# put a source in the first node
simulator.environment.node_list()[0].data["source"] = True
target = simulator.environment.node_list()[-1]
target.data["target"] = True
# schedule the main function
for node in simulator.environment.nodes.values():
    simulator.schedule_event(random.random() / 100, aggregate_program_runner, simulator, 1.1, node, main)
# render
CustomRenderMonitor(
    simulator,
    RenderConfig(
        effects=[CustomDrawEdges(), CustomDrawNodes(color_from="result")],
        mode=RenderMode.SAVE,
        save_as="output/scr.mp4",
        dt=0.1
    )
)
simulator.run(200)