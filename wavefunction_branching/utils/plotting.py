import matplotlib.pyplot as plt
import numpy as np


def vprint(string: str, verbose=False) -> None:
    if verbose:
        print(string)


def vis(varname_var, name="-", threshold=1e-5, figsize=(5, 5), output=None, output_folder=""):
    """Helper function to visualize a tensor with imshow().
    Input a dictionary, eg. vis({'tensor_name': tensor}), or vis(tensor, name='tensor_name')"""
    plt.rcParams["figure.figsize"] = figsize
    if not isinstance(varname_var, dict):
        varname_var = {name: varname_var}
    for varname, T_ in varname_var.items():
        T = np.array(T_).copy()
        if len(T.shape) <= 1:
            print(f"{varname} = {T}")
        else:
            if len(T.shape) > 2:
                axis = tuple(np.arange(len(T.shape) - 2, dtype=int))
                T = np.sum(abs(T), axis=axis)
            print(f"{varname} is a {len(T_.shape)}-leg {T_.dtype} tensor with shape {T_.shape}:")
            plt.imshow(np.log(abs(T) + threshold))
            if varname != "-":
                plt.title(f"log| {varname} |")
            if output or output_folder != "":
                if isinstance(output, str):
                    plt.savefig(output_folder + output)
                else:
                    plt.savefig(output_folder + varname + ".png")
            plt.show()
