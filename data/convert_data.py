import numpy as np

name = "task_948_bwa__DA45_.csv"
#memory_limit_in_mb time cpu_usage_in_pct io_read_bytes io_write_bytes memory_usage_in_pct memory_usage_in_mb max_cpus max_mem timestamp
data = np.loadtxt(name, delimiter=' ', usecols=(9, 2, 3 ,4, 6), skiprows=1).transpose() # dtype=np.double
X = list(np.r_[data[0] - data[0][0] + data[0][1] - data[0][0]])
Y = data[1:]

np.savetxt(name+"CPU.txt", np.array([X, Y[0]]).T)
np.savetxt(name+"MEM.txt", np.array([X, Y[3]]).T)
