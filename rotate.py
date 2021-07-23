import math

def rotate(origin, point, angle):

    ox = origin[0]
    oy = origin[1]
    px = point[:,0]
    py= point[:,1]
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy