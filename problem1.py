import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.cluster import KMeans
#1d)
sigmasq = 1
D = np.linspace(1e-6, 2*sigmasq,10000)
R = np.maximum(0.5*np.log(sigmasq/D),0)

Smax = np.sqrt(12*sigmasq)/2
bits = 2
delta = pow(2,bits)

#1e)
def uniform_quantizer(bits):
    sigmasq = 1
    Smax = np.sqrt(12 * sigmasq) / 2
    delta_ = pow(2, bits)
    delta = 2*Smax/delta_
    Dq = (pow(delta,2)/12)
    Rq = np.log2(2*Smax/delta)
    return [Dq, Rq]

quantizer_points_D = [uniform_quantizer(1)[0],
                    uniform_quantizer(2)[0],
                    uniform_quantizer(3)[0]]

quantizer_points_R = [uniform_quantizer(1)[1],
                    uniform_quantizer(2)[1],
                    uniform_quantizer(3)[1]]

fig, ax = plt.subplots()
ax.plot(D,R, "r")
ax.scatter(quantizer_points_D, quantizer_points_R)
ax.annotate("R=1", (quantizer_points_D[0], quantizer_points_R[0]))
ax.annotate("R=2", (quantizer_points_D[1], quantizer_points_R[1]))
ax.annotate("R=3", (quantizer_points_D[2], quantizer_points_R[2]))
ax.grid()
ax.set_title("Distortion-Rate")
ax.set_title("Distortion-Rate with uniform quantization rate points.")
ax.set_xlabel("Distortion D")
ax.set_ylabel("Rate R")
ax.legend(["Distortion-Rate R(D)", "Quantization rate points"])
plt.show()

#1f)
def non_uniform_quantizer(sigmaq):
    sigmasq = sigmaq
    Smax = np.sqrt(12 * sigmasq) / 2
    delta_ = pow(2, 2)
    delta = 2*Smax/delta_
    Dq = (pow(delta,2)/12)
    Rq = np.log2(2*Smax/delta)
    return [Dq, Rq]

quantizer_points_D = [uniform_quantizer(2)[0],
                    non_uniform_quantizer(0.30)[0]]

quantizer_points_R = [uniform_quantizer(2)[1],
                    non_uniform_quantizer(0.30)[1]]

fig, ax = plt.subplots()
ax.plot(D,R, "r")
ax.scatter(quantizer_points_D, quantizer_points_R)
ax.annotate("Uniform", (quantizer_points_D[0], quantizer_points_R[0]))
ax.annotate("Non-uniform", (quantizer_points_D[1], quantizer_points_R[1]))
ax.grid()
ax.set_title("Distortion-Rate with uniform quantization rate points.")
ax.set_xlabel("Distortion D")
ax.set_ylabel("Rate R")
ax.set_xlim(0,0.25)
ax.set_ylim(1,3)
ax.legend(["Distortion-Rate R(D)", "Quantization rate points"])
plt.show()


#1g)
def alpahbet(n,R):
    A = [[0]]
    for i in range(n*R):
        A.append([(pow(2,i))])
    return np.array(A)


X_1 = alpahbet(1,1)
X_2 = alpahbet(1,2)
X_3 = alpahbet(1,3)

s = np.random.normal(0, 1, 3)
print(s)

kmeans_1 = KMeans(n_clusters=2, random_state=0).fit(X_1)
kmeans_2 = KMeans(n_clusters=2, random_state=0).fit(X_2)
kmeans_3 = KMeans(n_clusters=2, random_state=0).fit(X_3)

cluster_point_1 = [kmeans_1.cluster_centers_[0], kmeans_1.cluster_centers_[1]]
cluster_point_2 = [kmeans_2.cluster_centers_[0], kmeans_2.cluster_centers_[1]]
cluster_point_3 = [kmeans_3.cluster_centers_[0], kmeans_3.cluster_centers_[1]]
mean_23 = (cluster_point_2[0] + cluster_point_3[0])/2

fig, ax = plt.subplots()
ax.plot(D,R, "r")
ax.scatter(cluster_point_1[0], cluster_point_1[1])
ax.scatter(cluster_point_2[0], cluster_point_2[1])
ax.scatter(mean_23, 3)
ax.annotate("R=1", (cluster_point_1[0], cluster_point_1[1]))
ax.annotate("R=2", (cluster_point_2[0], cluster_point_2[1]))
ax.annotate("R=3", (mean_23, 3))
ax.grid()
ax.set_title("Distortion-Rate")
ax.set_title("Distortion-Rate with vector quantizer.")
ax.set_xlabel("Distortion D")
ax.set_ylabel("Rate R")
ax.legend(["Distortion-Rate R(D)", "Vector Quanitzation points"])
plt.show()
