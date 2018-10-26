
from . pylibh2 import *


def new_square_macrosurface(ax, bx):

    ms = Macrosurface3d(4, 5, 2)

    #  vertices 
    #  bottom left 
    ms.x[0][0] = -ax / 2
    ms.x[0][1] = -bx / 2
    ms.x[0][2] = 0.0
    #  bottom right 
    ms.x[1][0] = ax / 2
    ms.x[1][1] = -bx / 2
    ms.x[1][2] = 0.0
    #  top right 
    ms.x[2][0] = ax / 2
    ms.x[2][1] = bx / 2
    ms.x[2][2] = 0.0
    #  top left 
    ms.x[3][0] = -ax / 2
    ms.x[3][1] = bx / 2
    ms.x[3][2] = 0.0

    #  vertex edges 
    #  bottom 
    ms.e[0][0] = 0
    ms.e[0][1] = 1
    #  right 
    ms.e[1][0] = 1
    ms.e[1][1] = 2
    #  top 
    ms.e[2][0] = 2
    ms.e[2][1] = 3
    #  left 
    ms.e[3][0] = 3
    ms.e[3][1] = 0
    #  diagonal 
    ms.e[4][0] = 1
    ms.e[4][1] = 3

    #  triangles and triangle edges 
    #  bottom left 
    ms.t[0][0] = 0
    ms.t[0][1] = 1
    ms.t[0][2] = 3
    ms.s[0][0] = 4
    ms.s[0][1] = 3
    ms.s[0][2] = 0
    #  top right 
    ms.t[1][0] = 1
    ms.t[1][1] = 2
    ms.t[1][2] = 3
    ms.s[1][0] = 2
    ms.s[1][1] = 4
    ms.s[1][2] = 1

    return ms