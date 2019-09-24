import numpy as np

def computeParams(a1, a2, a3, b):
    u0 = 1 / np.linalg.norm(a3.T) ** 2 * (a1.T.dot(a3))
    v0 = 1 / np.linalg.norm(a3.T) ** 2 * (a2.T.dot(a3))
    av = np.sqrt(1 / np.linalg.norm(a3.T) ** 2 * a2.T.dot(a2) - v0 ** 2)
    s = (1 / np.linalg.norm(a3.T) ** 4) / av * np.cross(a1.T, a3.T).dot(np.cross(a2.T, a3.T).T)
    compute = 1 / np.linalg.norm(a3.T) ** 2 * a1.T.dot(a1) - s ** 2 - u0 ** 2
    au = np.sqrt(compute)
    K = np.array([[au, s, u0],[0, av, v0],[0, 0, 1]])
    b1 = 1 / np.linalg.norm(a3.T) ** 2 / av
    b2 = np.cross(a2.T, a3.T)
    R = np.array([(b1 * b2).T, np.cross(np.sign(b[2]) * 1 / np.linalg.norm(a3.T) * a3, 1 / np.linalg.norm(a3.T) ** 2 / av * np.cross(a2.T, a3.T)).T, (np.sign(b[2]) * 1 / np.linalg.norm(a3.T) * a3).T])
    T = np.sign(b[2]) * 1 / np.linalg.norm(a3.T) * np.linalg.inv(K).dot(b).T
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

def meanSquareError(M, output, input):
    a = 0
    for i, j in zip(output, input):
        pi = np.concatenate([np.array(i), [1]])
        m1 = ((M[0][:4]).T.dot(pi))
        m2 = ((M[1][:4]).T.dot(pi))
        m3 = ((M[2][:4]).T.dot(pi))
        exi = m1 / m3
        eyi = m2 / m3
        a += ((j[0] - exi) ** 2 + (j[1] - eyi) ** 2)
    a = a / len(output)
    print("Mean Square Error = %s\n" % a)

def matirxA(op, ip):
    size = 4
    A = []
    zero = np.zeros(size)
    for i, j in zip(op, ip):
        pi = np.concatenate([np.array(i), [1]])
        xipi = j[0] * np.concatenate([np.array(i), [1]])
        yipi = j[1] * np.concatenate([np.array(i), [1]])
        A.append(np.concatenate([np.concatenate([np.array(i), [1]]), zero, -xipi]))
        A.append(np.concatenate([zero, np.concatenate([np.array(i), [1]]), -yipi]))
    # print(np.array(A))
    return np.array(A)

def matrixM(A):
    u, s, v = np.linalg.svd(A, full_matrices = True)
    M = v[-1].reshape(3, 4)
    b = []
    for i in range(len(M)):
        m = M[i][3]
        b.append(m)
    b = np.reshape(b, (3, 1))
    return M[0][:3].T, M[1][:3].T, M[2][:3].T, b, M

def main():
    output, input = [], []
    with open("data/points.txt") as file:
        points = file.readlines()
        for k in points:
            if k:
                point = k.split()
                p1 = [float(j) for j in point[:3]]
                p2 = [float(j) for j in point[3:]]
                output.append(p1)
                input.append(p2)
    print(__doc__)
    A = matirxA(output, input)
    a1, a2, a3, b, M = matrixM(A)
    computeParams(a1, a2, a3, b)
    meanSquareError(M,output,input)
