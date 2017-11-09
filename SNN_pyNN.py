import os
from six.moves import cPickle
import warnings
import pyNN.spiNNaker as sim
#from pyNN.utility import get_simulator, init_logging, normalized_filename
#import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt

MODEL = 'weights.02-0.49_brian'
PATH = './model/dv36_evtacc_D64_B0_30E'


def load_assembly(path, filename):
    """Load the populations in an assembly.

    Loads the populations in an assembly that was saved with the
    `save_assembly` function.

    The term "assembly" refers to pyNN internal nomenclature, where
    ``Assembly`` is a collection of layers (``Populations``), which in turn
    consist of a number of neurons (``cells``).

    Parameters
    ----------

    path: str
        Path to directory where to load model from.

    filename: str
        Name of file to load model from.

    Returns
    -------

    layers: list[pyNN.Population]
        List of pyNN ``Population`` objects.
    """

    import sys

    filepath = os.path.join(path, filename)
    assert os.path.isfile(filepath), \
        "Spiking neuron layers were not found at specified location."
    if sys.version_info < (3,):
        s = cPickle.load(open(filepath, 'rb'))
    else:
        s = cPickle.load(open(filepath, 'rb'), encoding='bytes')

    # Iterate over populations in assembly
    layers = []
    for label in s['labels']:
        celltype = getattr(sim, s[label]['celltype'])
        population = sim.Population(s[label]['size'], celltype,
                                         celltype.default_parameters,
                                         structure=s[label]['structure'],
                                         label=label)
        # Set the rest of the specified variables, if any.
        for variable in s['variables']:
            if getattr(population, variable, None) is None:
                setattr(population, variable, s[label][variable])
        if label != 'InputLayer':
            population.set(i_offset=s[label]['i_offset'])
        layers.append(population)

    return layers

def read_weights(filepath):
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



# def load(path, filename):
#
#     layers = load_assembly(path, filename)
#     for i in range(len(layers ) -1):
#         filepath = os.path.join(path, layers[ i +1].label)
#         assert os.path.isfile(filepath), \
#             "Connections were not found at specified location."
#         connections = read_weights(filepath)
#         connector = sim.FromListConnector(connections, column_names=["i", "j", "weight", "delay"])
#         proj_1 = sim.Projection(layers[i], layers[i+1], connector)

def load(path, filename):

    layers = load_assembly(path, filename)
    for i in range(len(layers ) -1):
        filepath = os.path.join(path, layers[ i +1].label)
        assert os.path.isfile(filepath), \
            "Connections were not found at specified location."
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.warn('deprecated', UserWarning)
            sim.Projection(layers[i], layers[ i +1],
                                sim.FromFileConnector(filepath))



sim.setup(timestep=1.0)
#sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
load(PATH, MODEL)

simtime = 10
sim.run(simtime)

sim.end()