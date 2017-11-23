import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import misc
import time
import numpy as np
import os



SIMTIME = 1000
BATCH_SIZE = 30
EVENTFRAME_WIDTH = None
MEASUREMENTS = False
INHIBITORY = False
output_spikes = []

if INHIBITORY:
    path = './model/dvs36_evtaccCOR_D16_B0_FLAT_30E/'
else:
    path = './model/dvs36_evtacc_D16_B0_FLAT_posW_10E/'
p1 = path + '01Dense_16'
p2 = path + '02Dense_4'


#filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/test/')
filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/balanced_100/')
#filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/three/')
#filepaths, labels = misc.get_sample_filepaths_and_labels('./data/aedat/one/')

sim.setup(timestep=1.0)

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

if MEASUREMENTS:
    pop_1.record(["spikes", "v"])
    pop_2.record(["spikes", "v"])
    pops = [pop_1, pop_2]
else:
    pop_2.record(["spikes"])
    pops = [pop_2]
#
# start = time.time()
# misc.run_testset(sim, SIMTIME, filepaths, labels, input_pop, pop_2, True)
# end = time.time()
# print(end - start)
# NR. OF SAMPLES: 400
# ACCURACY: 0.6175
# CLASS ACCURACIES N L C R: 0.92 0.6 0.54 0.41
# 2935.13000011 (batch 100)

#
# start = time.time()
# misc.run_testset_sequence(sim, SIMTIME, filepaths, labels, input_pop, pop_2, pops, True, 100, 10)
# end = time.time()
# print(end - start)
# 50.5559999943 seconds
# Application started - waiting 439.985 seconds for it to stop (balanced_100)


start = time.time()
misc.run_testset_sequence_in_batches(sim, SIMTIME, filepaths, labels, BATCH_SIZE, input_pop, pop_2, pops, True, 100, EVENTFRAME_WIDTH)
end = time.time()
print(end - start)
# NR. OF SAMPLES: 400
# ACCURACY: 0.675
# CLASS ACCURACIES N L C R: 0.93 0.66 0.62 0.49
# 732.963000059 (batchsize=30, vpot measurement etc)

# NR. OF SAMPLES: 400
# ACCURACY: 0.635
# CLASS ACCURACIES N L C R: 0.97 0.66 0.61 0.3
# 747.660000086 (batchsize=100, only output spikes measured..)
# 2017-11-22 15:34:33 WARNING: The reinjector on 0, 0 has detected that 6 packets were dumped from a core failing to take
# the packet. This often occurs when the executable has crashed or has not been given a multicast packet callback. It can
# also result from the core taking too long to process each packet. These packets were reinjected and so this number is
# likely a overestimate.
#
# NR. OF SAMPLES: 400
# ACCURACY: 0.55
# CLASS ACCURACIES N L C R: 0.93 0.59 0.54 0.28
# 234.315000057 (batch 50, simtime 250)
#
# NR. OF SAMPLES: 400
# ACCURACY: 0.515
# CLASS ACCURACIES N L C R: 0.93 0.59 0.54 nan
# 213.197000027 (batch 100, simtime 250)

# NR. OF SAMPLES: 400
# ACCURACY: 0.54
# CLASS ACCURACIES N L C R: 0.99 0.43 0.46 0.28
# 289.74000001 (batch 30, evtframe_width=10, simtime=250)
#
# NR. OF SAMPLES: 400
# ACCURACY: 0.5525
# CLASS ACCURACIES N L C R: 1.0 0.44 0.48 0.29
# 687.986999989 (batch 30, evtframe_width=10, simtime=1000)