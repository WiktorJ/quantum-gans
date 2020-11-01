import math


def map_to_radians(val, low=-1, high=1):
    span = high - low
    norm_val = float(val - low) / float(span)
    return norm_val * math.pi