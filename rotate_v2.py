import numpy as np

def rotate_v2(point, origin, degrees):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(point)
    rotated_coord=np.squeeze((R @ (p.T-o.T) + o.T).T)
    #rotated_coord=np.squeeze((R @ (p.T) ).T)
    return rotated_coord
