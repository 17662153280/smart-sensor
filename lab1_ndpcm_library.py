
from collections import namedtuple
import numpy as np

# Import quantizer for error
import lab1_library

# Declaring namedtuple()
# n - total length of the simulation (number of samples/iterations)
# h_depth - number of history elements in \phi and corresponding coefficients (length of vectors)
# n_bits - number of bits to be transmitted (resolution of encoded error value)
# phi - vector of vectors of samples history (reproduced!!) 
#       - first index = iteration; second index = current time vector element
# theta - vector of vectors of coefficients 
#       - first index = iteration; second index = current time vector element
# y_hat - vector of all predicted (from = theta * phi + k_v * eq)
# e - exact error between the sample and the predicted value (y_hat)
# eq - quantized value of error (see n_bits!!)
# y_recreated - vector of all recreated/regenerated samples (used in the prediction!!)
NDPCM = namedtuple('NDAPCM', ['n', 'h_depth', 'n_bits',
                   'phi', 'theta', 'y_hat', 'e', 'eq', 'y_recreated'])


def init(n, h_depth, n_bits):
    # Adding values
    data_block = NDPCM(
        n, h_depth, n_bits, np.zeros((n, h_depth)), np.zeros(
            (n, h_depth)), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    )
    data_block.phi[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    data_block.eq[0] = np.array([0])  # Initialize eq for the first iteration
    ### Modify initial value for any component, parameter:
    # ...
    return data_block


def prepare_params_for_prediction(data_bloc, k):
    # Update weights for next round (k) based on previous k-1, k-2,...
    # Initialize 'phi' and 'theta' for the first and second iterations
    if (k == 1):
        # Initialize phi and theta for the first iteration
        data_bloc.phi[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Initialize phi for the first iteration
        data_bloc.theta[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Initialize theta for the first iteration
        return
    if (k == 2):
        # Initialize phi and theta for the second iteration
        data_bloc.phi[1] = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Initialize phi for the second iteration
        data_bloc.theta[1] = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Initialize theta for the second iteration
        return
    # Fill 'phi' history for 'h_depth' last elements
    data_bloc.phi[k] = np.array([
        data_bloc.y_recreated[k],  # Add last recreated value (y(k-1))
        data_bloc.y_recreated[k-1], # Copy shifted from previous history (y(k-2))
        data_bloc.y_recreated[k-2],
        data_bloc.y_recreated[k-3],
        data_bloc.y_recreated[k-4],
        data_bloc.y_recreated[k-5],
        data_bloc.y_recreated[k-6],
        data_bloc.y_recreated[k-7]    
    ])
    print("e=", data_bloc.eq[k])
    print("eT=", data_bloc.eq[k].transpose())
    # TODO: Update weights/coefficients 'theta'

    return


def predict(data_bloc, k):
    if (k > 0):
        data_bloc.phi[k] = data_bloc.phi[k-1]
    # Calculate 'hat y(k)' based on (k-1) parameters
    data_bloc.y_hat[k] = np.dot(data_bloc.theta[k-1], data_bloc.phi[k-1])
    # If it's the first iteration, set y_hat[1] based on the input
    if (k == 1):
        data_bloc.y_hat[1] = data_bloc.phi[0][0]  # Use the first input value as initial prediction
    print(data_bloc.theta[k] @ data_bloc.phi[k])
    return data_bloc.phi[k][0]


def calculate_error(data_block, k, real_y):
    data_block.e[k] = real_y - data_block.y_hat[k]
    data_block.eq[k] = lab1_library.quantize(
        data_block.e[k], data_block.n_bits)
    return data_block.eq[k]


def reconstruct(data_block, k):
    # Update theta based on the current error and phi
    # For simplicity, let's just update theta with a simple linear regression for now
    if k > 1:
        X = np.vstack([data_block.phi[i] for i in range(1, k)])
        y = data_block.e[1:k]
        data_block.theta[k-1] = np.linalg.lstsq(X, y, rcond=None)[0]
    data_block.y_recreated[k] = data_block.y_hat[k] + data_block.eq[k]




