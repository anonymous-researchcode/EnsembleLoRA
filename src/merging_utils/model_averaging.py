import torch

def model_soup_averaging(state_dicts, scale=1.0):
    return_state_dict = {}
    for key in state_dicts[0].keys():
        return_state_dict[key] = torch.mean(torch.stack([state_dict[key] for state_dict in state_dicts], dim=0), dim=0)*scale

    return return_state_dict

def task_arithmetic_addition(state_dicts, scale=1.0):
    return_state_dict = {}
    for key in state_dicts[0].keys():
        return_state_dict[key] = torch.sum(torch.stack([state_dict[key] for state_dict in state_dicts], dim=0), dim=0)*scale

    return return_state_dict