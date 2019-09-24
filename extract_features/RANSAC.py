import numpy as np
import random
import math


def computeParams(M):
    bmatrix = []
    row = 3
    col = 1
    for i in range(len(M)):
        bmatrix.append(M[i][3])
    bmatrix = np.reshape(bmatrix, (row, col))
    np.set_printoptions(formatter={'float': "{0:.6f}".format})
    l1 = (1 / np.linalg.norm(M[2][:3].T.T)) ** 2
    l2 = (M[0][:3].T.T.dot(M[2][:3].T))
    u0 = l1 * l2
    w1 = (1 / np.linalg.norm(M[2][:3].T.T)) ** 2
    w2 = (M[1][:3].T.T.dot(M[2][:3].T))
    v0 = w1 * w2
    a22 = M[1][:3].T.T.dot(M[1][:3].T)
    av = np.sqrt((1 / np.linalg.norm(M[2][:3].T.T)) ** 2 * a22 - v0 ** 2)
    a1xa3 = np.cross(M[0][:3].T.T, M[2][:3].T.T)
    a2xa3 = np.cross(M[1][:3].T.T, M[2][:3].T.T)
    s = ((1 / np.linalg.norm(M[2][:3].T.T)) ** 4) / av * a1xa3.dot(a2xa3.T)
    a12 = M[0][:3].T.T.dot(M[0][:3].T)
    norm = (1 / np.linalg.norm(M[2][:3].T.T))
    compute = norm ** 2 * a12 - s ** 2 - u0 ** 2
    au = np.sqrt(compute)
    K = np.array([[au, s, u0], [0, av, v0], [0, 0, 1]])
    r1 = (1 / np.linalg.norm(M[2][:3].T.T)) ** 2 / av * np.cross(M[1][:3].T.T, M[2][:3].T.T)
    r3 = np.sign(bmatrix[2]) * (1 / np.linalg.norm(M[2][:3].T.T)) * M[2][:3].T
    r2 = np.cross(np.sign(bmatrix[2]) * (1 / np.linalg.norm(M[2][:3].T.T)) * M[2][:3].T, (1 / np.linalg.norm(M[2][:3].T.T)) ** 2 / av * np.cross(M[1][:3].T.T, M[2][:3].T.T))
    b1 = np.sign(bmatrix[2])
    b2 = (1 / np.linalg.norm(M[2][:3].T.T))
    b3 = np.linalg.inv(K).dot(bmatrix).T
    T = b1 * b2 * b3
    R = np.array([r1.T, r2.T, r3.T])
    print("--------------------------------------")
    print("u0, v0 = %f, %f\n" % (u0, v0))
    print("--------------------------------------")
    print("alphaU,alphaV = %f, %f\n" % (au, av))
    print("--------------------------------------")
    print("s = %f\n" % s)
    print("--------------------------------------")
    print("K* = %s\n" % K)
    print("--------------------------------------")
    print("T* = %s\n" % T)
    print("--------------------------------------")
    print("R* = %s\n" % R)
    print("--------------------------------------")


def ransac(op, ip, prob, nmin, nmax, kmax):
    w = 0.5
    k = kmax
    count = 0
    num = 0
    mDistance = np.median(distance(matrixM(matirxA(op, ip)), op, ip))
    np.random.seed(0)
    while (count < k and count < kmax):
        index = np.random.choice(len(op), random.randint(nmin, nmax))
        ranOp, ranIp = np.array(op)[index], np.array(ip)[index]
        M = matrixM(matirxA(ranOp, ranIp))
        d = distance(M, op, ip)
        liner = []
        for i, d in enumerate(d):
            ac_d = 1.5 * mDistance
            if d < ac_d:
                liner.append(i)
        if len(liner) >= num:
            num = len(liner)
            bestM = matrixM(matirxA(ranOp, ranIp))
        if not (w == 0):
            inl = float(len(liner))
            ipl = float(len(ip))
            w = inl / ipl
            i = float(math.log(1 - prob))
            j = np.absolute(math.log(1 - (w ** random.randint(nmin, nmax))))
            k = i / j
        count += 1;
    return num, bestM


def distance(M, op, ip):
    d = []
    for i, j in zip(op, ip):
        pi = np.append(np.array(i), 1)
        exi = (M[0][:4].T.dot(pi)) / (M[2][:4].T.dot(pi))
        eyi = (M[1][:4].T.dot(pi)) / (M[2][:4].T.dot(pi))
        xe = (j[0] - exi) ** 2
        ye = (j[1] - eyi) ** 2
        di = np.sqrt((xe + ye))
        d.append(di)
    return d

def matrixM(A):
    u, s, v = np.linalg.svd(A, full_matrices=True)
    M = v[-1].reshape(3, 4)
    return M

def matirxA(op, ip):
    A = []
    size = 4
    zero = np.zeros(size)
    for i, j in zip(op, ip):
        con = np.concatenate([np.array(i), [1]])
        A.append(np.concatenate([np.concatenate([np.array(i), [1]]), zero, -j[0] * con]))
        A.append(np.concatenate([zero, np.concatenate([np.array(i), [1]]), -j[1] * con]))
    return np.array(A)



def main(noise):
    configname = "RANSAC.config"
    output, input = [], []
    with open(noise) as file:
        points = file.readlines()
        for k in points:
            if k:
                point = k.split()
                p1 = [float(j) for j in point[:3]]
                p2 = [float(j) for j in point[3:]]
                output.append(p1)
                input.append(p2)
    print(__doc__)
    with open(configname, 'r') as configure:
        prob = float(configure.readline().split()[0])
        kmax = int(configure.readline().split()[0])
        nmin = int(configure.readline().split()[0])
        nmax = int(configure.readline().split()[0])
    inlinerNum, bestM = ransac(output, input, prob, nmin, nmax, kmax)
    computeParams(bestM)

