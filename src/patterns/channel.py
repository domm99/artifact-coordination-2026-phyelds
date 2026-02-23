import random
from phyelds.calculus import aggregate
from phyelds.simulator import Simulator
from phyelds.libraries.device import sense
from phyelds.libraries.collect import collect_or
from phyelds.libraries.spreading import distance_to
from CustomRenderMonitor import CustomRenderMonitor
from phyelds.simulator.deployments import deformed_lattice
from phyelds.libraries.distances import neighbors_distances
from CustomDrawings import CustomDrawNodes, CustomDrawEdges
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.effects import RenderConfig, RenderMode


random.seed(42)


@aggregate
def main():
    distances = neighbors_distances()
    target_distance = distance_to(sense("target"), distances)
    nodes_in_path = collect_or(target_distance, sense("source"))
    distance_from_path = distance_to(nodes_in_path, distances)
    channel = 1.0 if distance_from_path < 1.12 else 0.0
    return channel


simulator = Simulator()
# deformed lattice
simulator.environment.set_neighborhood_function(radius_neighborhood(1.12))
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
    simulator.schedule_event(0.0, aggregate_program_runner, simulator, 0.1, node, main)
# render
CustomRenderMonitor(
    simulator,
    RenderConfig(
        effects=[CustomDrawEdges(), CustomDrawNodes(color_from="result")],
        mode=RenderMode.SAVE,
        save_as="output/channel.mp4",
        dt=0.1
    )
)
simulator.run(10)