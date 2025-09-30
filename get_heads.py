"""
Functions for extracting information from .npy files that saved the results of ablations. 
Includes experimentation on metrics to measure localization, including KL Div and Wasserstein. 
"""
import numpy as np

def top_ten(heatmap, vals=True):
    flat_indices = np.argsort(heatmap, axis=None)[-10:][::-1]
    row_indices, col_indices = np.unravel_index(flat_indices, heatmap.shape)
    top_coords = list(zip(row_indices, col_indices))
    if vals:
        return [(x, round(heatmap[x], 4)) for x in top_coords]
    else:
        return top_coords


def bottom_ten(heatmap, vals=True):
    flat_indices = np.argsort(heatmap, axis=None)[:10]
    row_indices, col_indices = np.unravel_index(flat_indices, heatmap.shape)
    top_coords = list(zip(row_indices, col_indices))
    if vals:
        return [(x, round(heatmap[x], 4)) for x in top_coords]
    else:
        return top_coords

def metric(knockout, complement):
    from scipy.stats import entropy, wasserstein_distance
    from scipy.special import softmax
    diff = complement - knockout
    diff = diff.flatten()
    uniform_dist = [1/len(diff)] * len(diff)
    kl = entropy(softmax(diff), uniform_dist)
    point = [0] * len(diff)
    point[0] = 1
    wd = wasserstein_distance(diff, point)
    return kl, wd

model = 'qwen_7b'
task = 'string_reverse'
print(task)
knockout = np.load(f'{model}_{task}_42.npy')
complement = np.load(f'{model}_{task}_complement_42.npy')

print("kl: ", round(metric(knockout, complement)[0], 4))
print("wasserstein: ", round(metric(knockout, complement)[1], 4))
print("argmax(C-K): ", top_ten(complement-knockout))
