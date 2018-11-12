

from cmut_nonlinear_sim.mesh import *
import numpy as np
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':

    if len(sys.argv[1:]):
        refn = int(sys.argv[1])
    else:
        refn = 2

    s = square(2, 2, refn)
    c = circle(2, refn)

    plt.figure()

    vertices = np.asarray(s.vertices)
    edges = np.asarray(s.edges)
    plt.plot(vertices[:,0], vertices[:,1], '.')
    for e in edges:
        x1, y1, z1 = vertices[e[0], :]
        x2, y2, z2 = vertices[e[1], :]
        plt.plot([x1, x2], [y1, y2], 'b-')
    plt.axis('equal')

    plt.figure()
    vertices = np.asarray(c.vertices)
    edges = np.asarray(c.edges)
    plt.plot(vertices[:,0], vertices[:,1], '.')
    for e in edges:
        
        x1, y1, z1 = vertices[e[0], :]
        x2, y2, z2 = vertices[e[1], :]
        plt.plot([x1, x2], [y1, y2], 'b-')
    plt.axis('equal')
    plt.show()

