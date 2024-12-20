import numpy as np
import math

#code copied from: https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta), R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta), R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta):
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R


rot = np.load("odo_rotation_matrix.npy")
eul = rot2eul(rot)
eul_deg = eul*180/math.pi
rot2 = eul2rot(eul)

print("Rotational matrix:\n", rot)
print("Euler angles (Radians):", eul)
print("Euler angles (Degrees):", eul_deg)
print("Rotational matrix from euler:\n", rot2)
