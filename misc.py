from PyAedatTools.ImportAedat import ImportAedat
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import  numpy as np
import os
import time



def override_weights(filepath, w_last = True):
    with open(filepath) as f:
        with open('output', 'w') as f_out:
            for i, line in enumerate(f):
                if i==0:
                    f_out.write(line)
                    continue
                line = line.split()  # to deal with blank
                if line:  # lines (ie skip them)
                    if w_last:
                        f_out.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + str(5.0) + '\n')
                    else:
                        f_out.write(line[0] + "\t" + line[1] + "\t" + str(5.0) + "\t" + line[2] + '\n')


def read_connections(filepath):
    with open(filepath) as f:
        exitatory_connections = []
        inhibitory_connections = []
        for i, line in enumerate(f):
            if i==0:
                continue
            line = line.split()  # to deal with blank
            if line:  # lines (ie skip them)
                line = [float(i) for i in line]
                if line[3] < 0.0:
                    line[3] = -line[3]
                    inhibitory_connections.append(line)
                else:
                    exitatory_connections.append(line)
    return inhibitory_connections, exitatory_connections

def extract_spiketimes_from_aedat(filepath, aedadt_dim=(240,180), target_dim=(36,36), no_gaps=False, start_time=0, simtime=float('Inf')):
    # Create a dict with which to pass in the input parameters.
    aedat = {}
    aedat['importParams'] = {}
    aedat['info'] = {}
    aedat['importParams']['filePath'] = filepath

    aedat_data = ImportAedat(aedat)

    scale = [float(target_dim[0]) / aedadt_dim[0], float(target_dim[1]) / aedadt_dim[1]]
    spike_times = [[] for i in range(target_dim[0]*target_dim[1])]

    # find minimum timestamp
    min_time = float('Inf')
    max_time = -float('Inf')
    for t in aedat_data['data']['polarity']['timeStamp']:
        if t < min_time:
            min_time = t
        if t > max_time:
            max_time = t

    last_t = float('Inf')
    t_step = 0
    for t, x, y in zip(aedat_data['data']['polarity']['timeStamp'], aedat['data']['polarity']['x'],
                          aedat_data['data']['polarity']['y']):
        x = int(x * scale[0])
        y = int(y * scale[1])
        x = 36-1-x
        y = 36-1-y

        if no_gaps:
            spike_times[x * 36 + y].append(t_step + start_time)
            if t > last_t:
                t_step += 1
            last_t = t
            if t_step >= simtime:
                break
        else:
            spike_times[x * 36 + y].append(t - min_time + start_time)  # reshape: [36,36] -> [1296], subtract min_time s.t. time values start at 0
            if (t - min_time) >= simtime:
                break

        if simtime < max_time-min_time:
            duration = simtime
        else:
            duration = max_time - min_time
    return spike_times, duration

def generate_input_sample_spikes(filepaths, no_gaps, pause_between_samples, inp_dim, sim_time_per_sample):
    starttime = 0
    all_sample_spikes = [[] for i in range(inp_dim)]
    starttimes = [starttime]
    durations = []
    for i, path in enumerate(filepaths):
        spike_times, duration = extract_spiketimes_from_aedat(path, no_gaps=no_gaps, start_time=starttime, simtime=sim_time_per_sample)
        for neuron, times in enumerate(spike_times):
            all_sample_spikes[neuron] += times
        starttime += duration + pause_between_samples
        starttimes.append(starttime)
        durations.append(duration)
    tot_simtime = starttimes[-1] - pause_between_samples
    return all_sample_spikes, starttimes[:-1], tot_simtime, durations

def set_cell_params(pop, cellparams):
    pop.set(v_thresh=1)
    pop.set(v_reset=0)
    pop.set(v_rest=0)
    #pop.set(e_rev_E=10)
    #pop.set(e_rev_I=-10)
    pop.set(i_offset=0)
    pop.set(cm=0.09)
    pop.set(tau_m=1000)
    pop.set(tau_refrac=0)
    pop.set(tau_syn_E=0.01)
    pop.set(tau_syn_I=0.01)


def get_sample_filepaths_and_labels(dataset_path):
    # Count the number of samples and classes
    classes = [subdir for subdir in sorted(os.listdir(dataset_path))
               if os.path.isdir(os.path.join(dataset_path, subdir))]

    label_dict = label_dict = {"N": "0", "L": "1", "C": "2", "R": "3",}
    num_classes = len(label_dict)
    assert num_classes == len(classes), \
        "The number of classes provided by label_dict {} does not match " \
        "the number of subdirectories found in dataset_path {}.".format(
            label_dict, dataset_path)

    filenames = []
    labels = []
    num_samples = 0
    for subdir in classes:
        for fname in sorted(os.listdir(os.path.join(dataset_path, subdir))):
            is_valid = False
            for extension in {'aedat'}:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                labels.append(label_dict[subdir])
                filenames.append(os.path.join(dataset_path, subdir, fname))
                num_samples += 1
    labels = np.array(labels, 'int32')
    print("Found {} samples belonging to {} classes.".format(
        num_samples, num_classes))
    return filenames, labels

def run_testset(sim, simtime, filepaths, labels, in_pop, out_pop, no_gaps):
    nr_runs = len(filepaths)

    neos = []
    for i, path in enumerate(filepaths):
        print('SAMPLE {}/{}: {}'.format(i, nr_runs, path))

        spike_times, _ = extract_spiketimes_from_aedat(path, no_gaps=no_gaps)
        in_pop.set(spike_times=spike_times)

        sim.run(simtime)

        if i<nr_runs-1:
            neos.append(out_pop.get_data(variables=["spikes"]))
            sim.reset()

    neos.append(out_pop.get_data(variables=["spikes"]))
    sim.end()

    correct_class_predictions = [0, 0, 0, 0]
    nr_class_samples = [0, 0, 0, 0]
    i = 0
    for neo, label in zip(neos, labels):

        output_spike_counts = [len(spikes) for spikes in neo.segments[i].spiketrains]
        prediction = np.argmax(output_spike_counts)
        if prediction == label:
            correct_class_predictions[label] += 1
        nr_class_samples[label] += 1

        print("PREDICTION/GND_TRUTH {}: {}/{}".format(i, prediction, label))
        i += 1
    correct_predictions = np.sum(correct_class_predictions)
    print("ACCURACY: {}".format(float(correct_predictions)/i))
    class_accuracies = np.array(correct_class_predictions) / np.array(nr_class_samples, dtype=float)
    print("CLASS ACCURACIES N L C R: {} {} {} {}".format(class_accuracies[0], class_accuracies[1], class_accuracies[2], class_accuracies[3]))

def run_testset_sequence(sim, simtime, filepaths, labels, in_pop, out_pop, pops, no_gaps, pause_between_samples):
    nr_samples = len(filepaths)

    input_spikes, starttimes, tot_simtime, durations = generate_input_sample_spikes(filepaths, no_gaps, pause_between_samples, in_pop.size, simtime)

    in_pop.set(spike_times=input_spikes)

    sim.run(tot_simtime)
    #neo = out_pop.get_data(variables=
    neo = []
    spikes = []
    v = []
    for i in range(len(pops)):
        neo.append(pops[i].get_data(variables=["spikes", "v"]))
        spikes.append(neo[i].segments[0].spiketrains)
        v.append(neo[i].segments[0].filter(name='v')[0])
    sim.end()
    path = './results/{}/'.format(int(time.time()))

    for i in range(len(pops)):
        plot.Figure(
            # plot voltage for first ([0]) neuron
            plot.Panel(v[i], ylabel="Membrane potential (mV)",
                       data_labels=[pops[i].label], yticks=True, xlim=(0, tot_simtime)),
            # plot spikes (or in this case spike)
            plot.Panel(spikes[i], yticks=True, markersize=3, xlim=(0, tot_simtime)),
            title="Simple Example",
            annotations="Simulated with {}".format(sim.name())
        ).save(path + 'figure_{}.png'.format(i))

    np.savez(file='inputspikes', arr_0=spikes[0])
    np.savez(file='pot1', arr_0=v[0])
    np.savez(file='pot2', arr_0=v[1])


    correct_class_predictions = [0, 0, 0, 0]
    nr_class_samples = [0, 0, 0, 0]

    start_index = [0, 0, 0, 0]
    for i in range(nr_samples):
        output_spike_counts = [0, 0, 0, 0]
        for out_neuron, spiketrain in enumerate(neo[len(pops)-1].segments[0].spiketrains):
            if start_index[out_neuron] >= spiketrain.size-1:
                continue
            for k, spiketime in enumerate(spiketrain[start_index[out_neuron]:]):
                next_start_index = k + start_index[out_neuron]
                if spiketime < starttimes[i] + simtime:
                    output_spike_counts[out_neuron] += 1
                else:
                    break
            start_index[out_neuron] = next_start_index

        prediction = np.argmax(output_spike_counts)
        label = labels[i]
        if prediction == label:
            correct_class_predictions[label] += 1
        nr_class_samples[label] += 1

    correct_predictions = np.sum(correct_class_predictions)
    print('NR. OF SAMPLES: {}'.format(nr_samples))
    print("ACCURACY: {}".format(float(correct_predictions)/nr_samples))
    class_accuracies = np.array(correct_class_predictions) / np.array(nr_class_samples, dtype=float)
    print("CLASS ACCURACIES N L C R: {} {} {} {}".format(class_accuracies[0], class_accuracies[1], class_accuracies[2], class_accuracies[3]))


def run_testset_sequence_in_batches(sim, simtime, filepaths, labels, batch_size, in_pop, out_pop, pops, no_gaps, pause_between_samples):
    tot_nr_samples = len(filepaths)
    nr_batches = int(len(filepaths) / float(batch_size)) + 1
    batch_nr_samples = [batch_size for i in range(nr_batches-1)]
    batch_nr_samples.append(tot_nr_samples % batch_size)
    batch_starttimes = []
    batch_labels = []
    neos = []
    for i in range(nr_batches):
        print('RUNNING BATCH {}/{}'.format(i+1,nr_batches))
        end_idx = (i + 1) * batch_size
        # last smaller batch
        if end_idx >= len(filepaths):
            batch_paths = filepaths[i * batch_size:]
            batch_labels.append(labels[i * batch_size:])
        else:
            batch_paths = filepaths[i * batch_size:end_idx]
            batch_labels.append(labels[i * batch_size:end_idx])

        input_spikes, starttimes, tot_simtime, durations = generate_input_sample_spikes(batch_paths, no_gaps, pause_between_samples, in_pop.size, simtime)
        batch_starttimes.append(starttimes)

        in_pop.set(spike_times=input_spikes)

        sim.run(tot_simtime)
        if i < nr_batches-1:
            neos.append(out_pop.get_data(variables=["spikes"]))
            sim.reset()

    neos.append(out_pop.get_data(variables=["spikes"]))
    sim.end()


    nr_class_samples = [0, 0, 0, 0]
    correct_class_predictions = [0, 0, 0, 0]

    for batch in range(nr_batches):

        start_index = [0, 0, 0, 0]
        for i in range(batch_nr_samples[batch]):
            output_spike_counts = [0, 0, 0, 0]
            for out_neuron, spiketrain in enumerate(neos[batch].segments[batch].spiketrains):
                if start_index[out_neuron] >= spiketrain.size-1:
                    continue
                for k, spiketime in enumerate(spiketrain[start_index[out_neuron]:]):
                    next_start_index = k + start_index[out_neuron]
                    if spiketime < batch_starttimes[batch][i] + simtime:
                        output_spike_counts[out_neuron] += 1
                    else:
                        break
                start_index[out_neuron] = next_start_index

            prediction = np.argmax(output_spike_counts)
            label = batch_labels[batch][i]
            if prediction == label:
                correct_class_predictions[label] += 1
            nr_class_samples[label] += 1

    correct_predictions = np.sum(correct_class_predictions)
    print('NR. OF SAMPLES: {}'.format(tot_nr_samples))
    print("ACCURACY: {}".format(float(correct_predictions)/tot_nr_samples))
    class_accuracies = np.array(correct_class_predictions) / np.array(nr_class_samples, dtype=float)
    print("CLASS ACCURACIES N L C R: {} {} {} {}".format(class_accuracies[0], class_accuracies[1], class_accuracies[2], class_accuracies[3]))