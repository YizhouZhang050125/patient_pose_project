import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle ABC in degrees.
    a, b, c are (x, y) tuples.
    The angle is measured at point b.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm == 0 or bc_norm == 0:
        return None

    cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)