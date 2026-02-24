import random
import pandas as pd
from pathlib import Path
from learning.model import MLP
from FBFLClient import fbfl_client
from phyelds.simulator import Simulator
from utils import distribute_nodes_spatially
from TestSetEvalMonitor import TestSetEvalMonitor
from custom_exporter import federations_count_csv_exporter
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig
from ProFed.partitioner import download_dataset, split_train_validation, partition_to_subregions

def run_simulation(threshold, number_subregions, seed):

    NUMBER_OF_ROUNDS = 3

    simulator = Simulator()

    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    deformed_lattice(simulator, 3, 3, 1, 0.01)

    initial_model_params = MLP().state_dict()

    devices = len(simulator.environment.nodes.values())
    mapping_devices_area = distribute_nodes_spatially(devices, number_subregions)

    print(f'Number of devices: {devices}')
    print(mapping_devices_area)

    train_data, test_data = download_dataset('EMNIST')

    train_data, validation_data = split_train_validation(train_data, 0.8)
    print(f'Number of training samples: {len(train_data)}')
    environment = partition_to_subregions(train_data, validation_data, 'EMNIST','Hard', number_subregions, seed)
    test_data, _ = split_train_validation(test_data, 1.0)
    environment_test = partition_to_subregions(test_data, test_data, 'EMNIST', 'Hard', number_subregions, seed)

    mapping = {}

    for region_id, devices in mapping_devices_area.items():
        mapping_devices_data = environment.from_subregion_to_devices(region_id, len(devices))
        mapping_devices_data_test = environment_test.from_subregion_to_devices(region_id, len(devices))
        for device_index, data in mapping_devices_data.items():
            device_id = devices[device_index]
            test_subset, _ = mapping_devices_data_test[device_index]
            complete_data = data[0], data[1], test_subset
            mapping[device_id] = complete_data

    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(
            0.0,
            aggregate_program_runner,
            simulator,
            1.0,
            node,
            fbfl_client,
            data=mapping[node.id],
            initial_model_params=initial_model_params,
            threshold=threshold,
            regions=number_subregions,
            max_time=NUMBER_OF_ROUNDS,
            seed = seed)

    config = ExporterConfig('output/', f'federations_seed-{seed}_regions-{number_subregions}', [], [], 3)
    simulator.schedule_event(0.96, federations_count_csv_exporter, simulator, 1.0, config)
    config = ExporterConfig('output/', f'experiment_seed-{seed}_regions-{number_subregions}_', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
    simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.add_monitor(TestSetEvalMonitor(simulator))
    simulator.run(NUMBER_OF_ROUNDS)


# Hyper-parameters configuration
threshold = 40.0
areas = 3
seed = 42

df = pd.DataFrame(columns=['timestamp', 'experiment'])

random.seed(seed)
print(f'Starting simulation with seed={seed}, regions={areas}, threshold={threshold}')
run_simulation(threshold, areas, seed)
