
import numpy as np

a = [[0.2, 0.8], [0.5, 0.5]]

N = np.zeros(shape=(2, 2), dtype=np.int)

p = [0]
num = 10000
for i in range(num):
    num = np.random.choice(a=[0, 1], p=a[p[-1]], size=1)[0]
    p.append(num)

    N[p[-2], p[-1]] += 1


print(N)

# Initialize B, T
B = [0.7,  0.3]
T = [0.5, 0.5]
eta = 0.0001

# Update
num_of_iters = 150
coeff = np.zeros(shape=(2, 2), dtype=np.float)
for iter in range(num_of_iters):
    for s in range(2):
        sum_st = 0.0
        for t in range(2):
            sum_st += np.exp(B[s] * T[t])
            #coeff[s, t] = counts[s, t] * B[s]


        for t in range(2):
            delta_t = 0.0
            for t_check in range(2):
                cc = 0.0
                if t_check == t:
                    cc = 1.0
                    delta_t += B[s] * float(N[s, t])*(1.0*cc - np.exp(B[s]*T[t])/sum_st)
            T[t] = T[t] + eta * delta_t

        for t in range(2):
            delta_s = 0.0
            for t_check in range(2):
                cc = 0.0
                if t_check == t:
                    cc = 1.0
                    delta_s += T[t] * float(N[s, t]) * (1.0*cc - np.exp(B[s] * T[t]) / sum_st)

            B[s] = B[s] + eta * delta_s




        """
        print(B)
        print(T)
        print("---------------")
        """
    print(B)

    E = np.zeros(shape=(2,2), dtype=np.float)
    for i in range(2):
        for j in range(2):
            E[i, j] = np.exp(B[i]*T[j])
        E[i, :] = E[i, :] / np.sum(E[i, :])

    print(E)
    print("---------------")