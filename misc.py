from PyAedatTools.ImportAedat import ImportAedat
import  numpy as np
import os

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
        connections = []
        for i, line in enumerate(f):
            if i==0:
                continue
            line = line.split()  # to deal with blank
            if line:  # lines (ie skip them)
                line = [float(i) for i in line]
                connections.append(line)
    return connections

def extract_spiketimes_from_aedat(filepath, aedadt_dim=(240,180), target_dim=(36,36)):
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

    for t, x, y in zip(aedat_data['data']['polarity']['timeStamp'], aedat['data']['polarity']['x'],
                          aedat_data['data']['polarity']['y']):
        x = int(x * scale[0])
        y = int(y * scale[1])
        x = 36-1-x
        y = 36-1-y
        spike_times[x * 36 + y].append(t - min_time)  # reshape: [36,36] -> [1296], subtract min_time s.t. time values start at 0

    return spike_times, max_time-min_time


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

def run_testset(sim, simtime, filepaths, labels, in_pop, out_pop):
    nr_runs = len(filepaths)

    neos = []
    for i, path in enumerate(filepaths):
        print('SAMPLE {}/{}: {}'.format(i, nr_runs, path))

        spike_times, _ = extract_spiketimes_from_aedat(path)
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




    


