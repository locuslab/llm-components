import numpy as np
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()
    f = np.load(args.filename)
    print(bottom_ten(f))
