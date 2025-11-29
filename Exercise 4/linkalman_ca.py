from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from linkalman import LinKalman
        
if __name__ == "__main__":
	# Add all necessary definitions and create kalman here.
	
    true_s = 100
    true_v = 1
    true_a = 0.1
    
    s_arr = []
    v_arr = []
    a_arr = []
    Ps_arr = []
    Pv_arr = []
    Pa_arr = []
    z_arr = []
    step_arr = []
    for step in range(1000):
        s_arr.append(kalman.x[0])
        v_arr.append(kalman.x[1])
        a_arr.append(kalman.x[2])

        Ps_arr.append(kalman.P[0][0])
        Pv_arr.append(kalman.P[1][1])
        Pa_arr.append(kalman.P[2][2])
        step_arr.append(step)
        print("Step ", step)
        print("  Predict State: ", kalman.predictState())
        print("  Predict Measurement: ", kalman.predictMeasurement())
        print("  K: ", kalman.computeKalmanGain())
        
        true_s = true_s + true_v + 0.5*true_a
        true_v = true_v + true_a
        
        print("true_s: ", true_s)
        print("true_v: ", true_v)
        print("true_a: ", true_a)
        z_mean = np.array([true_s])
        z = np.random.multivariate_normal(z_mean, R)
        print("  Update: ", kalman.update(z))
        z_arr.append(z[0])
    fig, axs = plt.subplots(4)
    axs[0].plot(step_arr, s_arr)[0].set_label("pos")
    axs[0].plot(step_arr, z_arr, "rx")[0].set_label("meas")
    axs[0].legend()
    axs[1].plot(step_arr, v_arr)[0].set_label("vel")
    axs[1].legend()
    axs[2].plot(step_arr, a_arr)[0].set_label("acc")
    axs[2].legend()
    axs[3].plot(step_arr, Ps_arr)[0].set_label("P_s")
    axs[3].plot(step_arr, Pv_arr)[0].set_label("P_v")
    axs[3].plot(step_arr, Pa_arr)[0].set_label("P_a")
    axs[3].legend()
    plt.show()
