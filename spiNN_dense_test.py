import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import misc
import time
import numpy as np
import os



SIMTIME = 1000
INHIBITORY = False
output_spikes = []

if INHIBITORY:
    path = './model/dvs36_evtaccCOR_D16_B0_FLAT_30E/'
else:
    path = './model/dvs36_evtacc_D16_B0_FLAT_posW_10E/'
p1 = path + '01Dense_16'
p2 = path + '02Dense_4'


filepath1 =  './data/aedat/' + 'rec_10_sample_2775_N.aedat'
filepath2 =  './data/aedat/' + 'rec_10_sample_535_C.aedat'
filepath3 =  './data/aedat/' + 'rec_10_sample_3112_L.aedat'
filepath4 =  './data/aedat/' + 'rec_10_sample_3248_L.aedat'
filepath4 =  './data/aedat/' + 'test_dvs_6.aedat'

#filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/test/')
filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/three/')
# filepaths = [filepath1, filepath2, filepath3, filepath4]
# labels = [0,2,1,1]
# filepaths = [filepath4]
# labels = [0]

sim.setup(timestep=1.0)

spike_times, _ = misc.extract_spiketimes_from_aedat(filepath1)

input_pop = sim.Population(size=1296, cellclass=sim.SpikeSourceArray(spike_times=[]), label="spikes")
pop_1 = sim.Population(size=16, cellclass=sim.IF_curr_exp(), label="1_input")
if INHIBITORY:
    pop_1.set(v_thresh=0.05)
else:
    pop_1.set(v_thresh=0.1)
pop_2 = sim.Population(size=4, cellclass=sim.IF_curr_exp(), label="2_hidden")
pop_2.set(v_thresh=0.1)

# inhibitory_connections = Projection(pre, post,
#                                     connector=sparse_connectivity,
#                                     synapse_type=facilitating,
#                                     receptor_type='inhibitory',
#                                     space=space,
#                                     label="inhibitory connections")

if INHIBITORY:
    inhibitory_connections_1, exitatory_connections_1 = misc.read_connections(p1)
    inhibitory_connector_1 = sim.FromListConnector(inhibitory_connections_1, column_names=["i", "j", "delay", "weight"])
    exitatory_connector_1 = sim.FromListConnector(exitatory_connections_1, column_names=["i", "j", "delay", "weight"])
    inhibitory_proj_1 = sim.Projection(input_pop, pop_1, inhibitory_connector_1, receptor_type='inhibitory')
    exitatory_proj_1 = sim.Projection(input_pop, pop_1, exitatory_connector_1, receptor_type='excitatory')

    inhibitory_connections_2, exitatory_connections_2 = misc.read_connections(p2)
    inhibitory_connector_2 = sim.FromListConnector(inhibitory_connections_2, column_names=["i", "j", "delay", "weight"])
    exitatory_connector_2 = sim.FromListConnector(exitatory_connections_2, column_names=["i", "j", "delay", "weight"])
    inhibitory_proj_2 = sim.Projection(pop_1, pop_2, inhibitory_connector_2, receptor_type='inhibitory')
    exitatory_proj_2 = sim.Projection(pop_1, pop_2, exitatory_connector_2, receptor_type='excitatory')

else:
    _, connections_1 =  misc.read_connections(p1)
    connector_1 = sim.FromListConnector(connections_1, column_names=["i", "j", "delay", "weight"])
    proj_1 = sim.Projection(input_pop, pop_1, connector_1)

    _, connections_2 = misc.read_connections(p2)
    connector_2 = sim.FromListConnector(connections_2, column_names=["i", "j", "delay", "weight"])
    proj_2 = sim.Projection(pop_1, pop_2, connector_2)

pop_2.record(["spikes"])
pops = [pop_1, pop_2]
#misc.run_testset(sim, SIMTIME, filepaths, labels, input_pop, pop_2, True)

misc.run_testset_sequence(sim, SIMTIME, filepaths, labels, input_pop, pop_2, pops, True, 1000)