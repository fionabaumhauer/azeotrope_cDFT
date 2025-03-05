import numpy as np
import tensorflow as tf
import sys
sys.path.append("/scratch/fb590/code/ion-cdft/cdft")
from scipy.integrate import simpson

# Enable or disable Tensor Float 32 Execution
tf.config.experimental.enable_tensor_float_32_execution(False)
import matplotlib.pyplot as plt


params = {"axes.labelsize": 14,
          "axes.titlesize": 16,}
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)

def place(ax, fig):
  ax.tick_params(direction="in", which="minor", length=3)
  ax.tick_params(direction="in", which="major", length=5, labelsize=13)
  ax.grid(which="major", ls="dashed", dashes=(1, 3), lw=0.8, zorder=0)
  #ax.legend(frameon=True, loc="best", fontsize=12,edgecolor="black")
  fig.tight_layout()

def generate_windows(array, bins):
    """
    Generate sliding windows for the input array with a given bin size.

    Parameters:
    - array (np.ndarray): Input array.
    - bins (int): Number of bins on each side of the central bin.
    - mode (str): Padding mode for np.pad (default is "wrap").

    Returns:
    - np.ndarray: Array of sliding windows.
    """
    padded_array = np.pad(array, bins, mode="wrap")
    windows = np.empty((len(array), 2 * bins + 1))
    for i in range(len(array)):
        windows[i] = padded_array[i:i + 2 * bins + 1]
    return windows


def c1_twotype(model_H, model_O, rho_H, rho_O, input_bins, dx, return_c2=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    window_bins = (input_bins - 1) // 2
    rhoH_windows = generate_windows(rho_H, window_bins).reshape(rho_H.shape[0], input_bins, 1)
    rhoO_windows = generate_windows(rho_O, window_bins).reshape(rho_O.shape[0], input_bins, 1)
    
    if return_c2:
        rhoH_windows = tf.Variable(rhoH_windows)
        rhoO_windows = tf.Variable(rhoO_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rhoO_windows)
            tape.watch(rhoH_windows)
            H_result = model_H(rhoH_windows, rhoO_windows)
            O_result = model_O(rhoH_windows, rhoO_windows)
        jacobi_windows_HH = tape.batch_jacobian(H_result, rhoH_windows).numpy().squeeze() / dx
        jacobi_windows_HO = tape.batch_jacobian(H_result, rhoO_windows).numpy().squeeze() / dx
        jacobi_windows_OO = tape.batch_jacobian(O_result, rhoO_windows).numpy().squeeze() / dx
        jacobi_windows_OH = tape.batch_jacobian(O_result, rhoH_windows).numpy().squeeze() / dx
       
        c1H_result = H_result.numpy().flatten()
        c1O_result = O_result.numpy().flatten()
        
        if return_c2 == "unstacked":
            return c1H_result, c1O_result, jacobi_windows_HH, jacobi_windows_HO, jacobi_windows_OO, jacobi_windows_OH
        
        c2_result_HH = np.row_stack([
            np.roll(np.pad(jacobi_windows_HH[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        c2_result_HO = np.row_stack([
            np.roll(np.pad(jacobi_windows_HO[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        c2_result_OO = np.row_stack([
            np.roll(np.pad(jacobi_windows_OO[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        c2_result_OH = np.row_stack([
            np.roll(np.pad(jacobi_windows_OH[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        
        return (c1H_result, c2_result_HH, c2_result_HO), (c1O_result, c2_result_OO, c2_result_OH)


   # if output_dict:
   #     c1H_result = model_H.predict_on_batch([rhoH_windows, rhoO_windows])["c1_H"].flatten()
   #     c1O_result = model_O.predict_on_batch([rhoO_windows, rhoH_windows])["c1_O"].flatten()
   # else:
   #     c1H_result = model_H.predict_on_batch([rhoH_windows, rhoO_windows]).flatten()
#        c1O_result = model_O.predict_on_batch([rhoO_windows, rhoH_windows]).flatten()

    c1H_result = model_H.predict_on_batch([rhoH_windows, rhoO_windows]).flatten()
    c1O_result = model_O.predict_on_batch([rhoO_windows, rhoH_windows]).flatten()
    return c1H_result, c1O_result


def betaFexc_twotype(model_H, model_O, rho_H, rho_O, input_bins, dx):
    """
    Calculate the excess free energy Fexc for a given density profile with functional line integration.

    model: The neural correlation functional
    rho: The density profile
    dx: The discretization of the input layer of the model
    """
    alphas = np.linspace(0, 1, 30)
    integrands = np.empty_like(alphas)
    
    for i, alpha in enumerate(alphas):
        
        c1H, c1O = c1_twotype(model_H, model_O, alpha * rho_H, alpha * rho_O, input_bins, dx)
        
        # simple rectangle rule as we have only one point
        # (Simpson's rule requires at least two points
        # to compute an integral because it uses quadratic interpolation.)
        integrands[i] = np.sum(rho_H * c1H + rho_O * c1O) * dx
    
    Fexc = -simpson(integrands, x=alphas)
    
    return Fexc

def get_P_range(rho_range, ratio):
    Fexc_range = np.empty_like(rho_range)
    derivPhiH = np.empty_like(rho_range)
    derivPhiO = np.empty_like(rho_range)
    z_range = np.ones(1)

    for i in range(len(rho_range)):
        
        rhoH_array = z_range * rho_range[i] * ratio
        rhoO_array = z_range * rho_range[i] * (1-ratio)
        
        Fexc_range[i] = kB * T * betaFexc_twotype(model_H, model_O, rhoH_array, rhoO_array, dx)
        c1H, c1O, = c1_twotype(model_H, model_O, rhoH_array, rhoO_array,input_bins,dx)
        derivPhiH[i] = -np.mean(c1H) * kB * T
        derivPhiO[i] = -np.mean(c1O) * kB * T

    P_range = (derivPhiH + kB*T) * rho_range * ratio + (derivPhiO + kB*T) * rho_range * (1-ratio) - Fexc_range/dx
    return P_range

def get_betamu(rho,c1):
    betamu = np.log(rho) - c1[0]
    return betamu