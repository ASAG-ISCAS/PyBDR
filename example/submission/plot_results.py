import numpy as np
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.util.visualization import plot

name = 'synchronous_16_20240116_154622'
with open('.\\results\\' + name + '.txt', 'r') as file:
    lines = file.readlines()

all_results = []
zono = []
current_zono = []

for line in lines:
    if line.startswith('Result:'):
        if zono:
            all_results.append(zono)
        zono = []
    else:
        if line.strip():
            current_zono.append([float(x) for x in line.split()])
        dim = 0
        if current_zono:
            dim = len(current_zono[0])
        if len(current_zono) == dim + 1:
            gen = []
            for di in range(1, dim+1):
                gen.append(current_zono[di])
            z = Zonotope(current_zono[0], np.array(gen))
            current_zono = []
            zono.append(z)
all_results.append(zono)
# all_results = [list(group) for group in zip(*all_results)]

print("Start drawing")
plot(all_results, [0, 1])
