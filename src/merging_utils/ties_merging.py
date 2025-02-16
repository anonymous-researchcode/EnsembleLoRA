import torch
import numpy as np
from merging_utils.ties_merging_utils import state_dict_to_vector, vector_to_state_dict, merge_methods


def resolve_lambda_code(lambda_code):
    if type(lambda_code) is tuple:
        lambda_list = torch.tensor(lambda_code)
    elif isinstance(lambda_code, float) or isinstance(lambda_code, int):
        lambda_list = torch.tensor([lambda_code])
    elif "linear+" in lambda_code:
        search_lambda, start, end, step = lambda_code.split("+")
        lambda_list = np.arange(eval(start), eval(end), eval(step))
    elif "mergelist" in lambda_code:
        task_lambdas = lambda_code.split("+")[-1].split(",")
        lambda_list = np.array(task_lambdas).astype(float).tolist()
    else:
        raise NotImplementedError(f"Unable to decode lambda_code {lambda_code}")
    return lambda_list

def ties_merging(
    state_dicts, 
    merge_function = "topk0.7_mass_dis-mean", # ${redundant}_${elect}_${agg}
    scale = 1.0
):
    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(check) for check in state_dicts]
    )

    reset, resolve, merge = merge_function.split("_")
    if "topk" in reset:
        reset_type = "topk"
        reset_thresh = eval(reset[len(reset_type) :])
    elif "std" in reset:
        reset_type = "std"
        reset_thresh = eval(reset[len(reset_type) :])
    elif "nf" in reset:
        reset_type = "nf"
        reset_thresh = eval(reset[len(reset_type) :])
    else:
        reset_type = ""
        reset_thresh = "none"

    merged_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )

    # lambdas = resolve_lambda_code(lambda_code)
    # assert len(lambdas) == 1
    # for lam in lambdas:
    # round(lam, 1)
    lam = scale
    print(
        f"Performing PRESS Merging with {merge_function} and Lambda {lam:.1f}"
    )
    merged_check = lam * merged_tv
    merged_checkpoint = vector_to_state_dict(
        merged_check, state_dicts[0]
    )
    return merged_checkpoint