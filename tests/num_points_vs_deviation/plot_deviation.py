import numpy as np
import matplotlib.pyplot as plt

def plot_results(results_dicts, function_labels, description):
    fig = plt.figure()
    fig.suptitle(description, fontsize = "10")
    axis1 = fig.add_subplot(121)
    axis1.set_ylabel('deviation in %')
    axis1.set_xlabel('number of gridpoints')
    axis1.set_ylim(0, 100)
    for i in range(len(function_labels)):
        axis1.plot(results_dicts[i].keys(), results_dicts[i].values(), marker =".",label = function_labels[i])
    axis1.legend() 

    axis2 = fig.add_subplot(122)
    axis2.set_ylabel('lg(deviation)')
    axis2.set_xlabel('number of gridpoints')
    axis2.set_yscale("log")
    axis2.set_ylim(10**-4, 1)
    for i in range(len(function_labels)):
        axis2.plot(results_dicts[i].keys(), 1/100*np.array(list(results_dicts[i].values())), marker =".",label = function_labels[i])
    axis2.legend()
    plt.show()