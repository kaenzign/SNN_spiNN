from PyAedatTools.ImportAedat import ImportAedat

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
    


