
import numpy as np

a = [[0.25, 0.75, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.0, 0.38, 0.25, 0.37], [0.0, 0.38, 0.37, 0.25]]
num_of_nodes = len(a)


N = np.zeros(shape=(num_of_nodes, num_of_nodes), dtype=np.int)

p = [0]
num = 100000
for i in range(num):
    num = np.random.choice(a=range(num_of_nodes), p=a[p[-1]], size=1)[0]
    p.append(num)

    N[p[-2], p[-1]] += 1


print(N)

# Initialize B, T
#B = [0.7,  0.3]
#T = [0.5, 0.5]

B = np.abs(np.random.normal(size=num_of_nodes))
T = np.abs(np.random.normal(size=num_of_nodes))

B = B / np.sum(B)
T = T / np.sum(T)


eta = 0.00001

# Update
num_of_iters = 250
#coeff = np.zeros(shape=(2, 2), dtype=np.float)
for iter in range(num_of_iters):
    for s in range(num_of_nodes):
        sum_st = 0.0
        for t in range(num_of_nodes):
            sum_st += np.exp(B[s] * T[t])
            #coeff[s, t] = counts[s, t] * B[s]


        for t in range(num_of_nodes):
            delta_t = 0.0
            for t_check in range(num_of_nodes):
                cc = 0.0
                if t_check == t:
                    cc = 1.0
                    delta_t += B[s] * float(N[s, t])*(1.0*cc - np.exp(B[s]*T[t])/sum_st)
            T[t] = T[t] + eta * delta_t

        for t in range(num_of_nodes):
            delta_s = 0.0
            for t_check in range(num_of_nodes):
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
    #print(B)

    E = np.zeros(shape=(num_of_nodes, num_of_nodes), dtype=np.float)
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            E[i, j] = np.exp(B[i]*T[j])
        E[i, :] = E[i, :] / np.sum(E[i, :])

    print(E)
    print("---------------")