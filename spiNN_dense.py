import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import misc
import time
import numpy as np
import os

INHIBITORY = False
SIMTIME = 250
path = './model/dvs36_evtacc_D16_B0_FLAT_posW_10E/'
#path = './model/dvs36_evtaccCOR_D16_B0_FLAT_30E/'
#path = './connections/'
p1 = path + '01Dense_16'
p2 = path + '02Dense_4'

NR = 1296

cellparams = {'v_thresh': 1,
                  'v_reset': 0,
                  'v_rest': 0,
                  'e_rev_E': 10,
                  'e_rev_I': -10,
                  'i_offset': 0,
                  'cm': 0.09,
                  'tau_m': 1000,
                  'tau_refrac': 0,
                  'tau_syn_E': 0.01,
                  'tau_syn_I': 0.01}

#filepath =  './data/aedat/' + 'rec_10_sample_3112_L.aedat'
# filepath =  './data/aedat/' + 'rec_10_sample_3248_L.aedat'
# filepath =  './data/aedat/' + 'rec_10_sample_1034_R.aedat'
# filepath =  './data/aedat/' + 'rec_10_sample_2194_R.aedat'
# filepath =  './data/aedat/' + 'rec_10_sample_0_N.aedat'
# filepath =  './data/aedat/' + 'rec_10_sample_2775_N.aedat'
filepath =  './data/aedat/' + 'rec_10_sample_535_C.aedat'
#filepath =  './data/aedat/' + 'test_dvs_6.aedat'

spike_times, simtime = misc.extract_spiketimes_from_aedat(filepath, no_gaps=True, eventframe_width=10)

sim.setup(timestep=1.0)

#first = [i for i in range(1000)]
#spike_times = [[i] for i in range(NR)]
#spike_times[0] = first
input_pop = sim.Population(size=NR, cellclass=sim.SpikeSourceArray(spike_times=spike_times), label="spikes")
pop_0 = sim.Population(size=NR, cellclass=sim.IF_curr_exp(), label="1_pre_input")
pop_0.set(v_thresh=0.1)
pop_1 = sim.Population(size=16, cellclass=sim.IF_curr_exp(), label="1_input")
pop_1.set(v_thresh=0.05)
#misc.set_cell_params(pop_1, cellparams)
pop_2 = sim.Population(size=4, cellclass=sim.IF_curr_exp(), label="2_hidden")
pop_2.set(v_thresh=0.1)
#misc.set_cell_params(pop_2, cellparams)
#pop_2.set(i_offset=s[label]['v_thresh'])

input_proj = sim.Projection(input_pop, pop_0, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=3, delay=1))

if INHIBITORY:
    inhibitory_connections_1, exitatory_connections_1 = misc.read_connections(p1)
    inhibitory_connector_1 = sim.FromListConnector(inhibitory_connections_1, column_names=["i", "j", "delay", "weight"])
    exitatory_connector_1 = sim.FromListConnector(exitatory_connections_1, column_names=["i", "j", "delay", "weight"])
    inhibitory_proj_1 = sim.Projection(pop_0, pop_1, inhibitory_connector_1, receptor_type='inhibitory')
    exitatory_proj_1 = sim.Projection(pop_0, pop_1, exitatory_connector_1, receptor_type='excitatory')

    inhibitory_connections_2, exitatory_connections_2 = misc.read_connections(p2)
    inhibitory_connector_2 = sim.FromListConnector(inhibitory_connections_2, column_names=["i", "j", "delay", "weight"])
    exitatory_connector_2 = sim.FromListConnector(exitatory_connections_2, column_names=["i", "j", "delay", "weight"])
    inhibitory_proj_2 = sim.Projection(pop_1, pop_2, inhibitory_connector_2, receptor_type='inhibitory')
    exitatory_proj_2 = sim.Projection(pop_1, pop_2, exitatory_connector_2, receptor_type='excitatory')
else:
    _, connections_1 = misc.read_connections(p1)
    #connector_1 = sim.FromFileConnector(p1)
    connector_1 = sim.FromListConnector(connections_1, column_names=["i", "j", "delay", "weight"])
    proj_1 = sim.Projection(pop_0, pop_1, connector_1)

    _, connections_2 = misc.read_connections(p2)
    #connector_2 = sim.FromFileConnector(p2)
    connector_2 = sim.FromListConnector(connections_2, column_names=["i", "j", "delay", "weight"])
    proj_2 = sim.Projection(pop_1, pop_2, connector_2)


pop_0.record(["spikes", "v"])
pop_1.record(["spikes", "v"])
pop_2.record(["spikes", "v"])
#pop_3.record(["spikes", "v"])

pops = [pop_0, pop_1, pop_2]


sim.run(SIMTIME)

neo = []
spikes = []
v = []
for i in range(len(pops)):
    neo.append(pops[i].get_data(variables=["spikes", "v"]))
    spikes.append(neo[i].segments[0].spiketrains)
#print(spikes)

    v.append(neo[i].segments[0].filter(name='v')[0])
#print (v)


sim.end()


path = './results/{}/'.format(int(time.time()))

for i in range(len(pops)):
    plot.Figure(
    # plot voltage for first ([0]) neuron
    plot.Panel(v[i], ylabel="Membrane potential (mV)",
    data_labels=[pops[i].label], yticks=True, xlim=(0, SIMTIME)),
    # plot spikes (or in this case spike)
    plot.Panel(spikes[i], yticks=True, markersize=3, xlim=(0, SIMTIME)),
    title="Simple Example",
    annotations="Simulated with {}".format(sim.name())
    ).save(path + 'figure_{}.png'.format(i))

plt.show()


np.savez(file='inputspikes', arr_0=spikes[0])
np.savez(file='pot1', arr_0=v[1])
np.savez(file='pot2', arr_0=v[2])

output_spikes = spikes[2]
output_spike_counts = [output_spikes[i].size for i in range(len(output_spikes))]
prediction = np.argmax(output_spike_counts)

print("PREDICTION: {}".format(prediction))


