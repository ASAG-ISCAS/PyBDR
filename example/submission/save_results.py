import numpy as np
from datetime import datetime

def save_result(R_list, name):
    currentDateTime = datetime.now().strftime('%Y%m%d_%H%M%S')

    fileName = f'./results/{name}_{currentDateTime}.txt'

    with open(fileName, 'w') as fileID:
        fileID.write('Result:\n')

        for ri_index in range(len(R_list)):
            ri = R_list[ri_index]

            for zi in ri:
                dim = len(zi.c)
                for di in range(dim):
                    fileID.write(f'{zi.c[di]} ')
                fileID.write('\n')

                for di in range(dim):
                    for j in range(len(zi.gen[0])):
                        fileID.write(f'{zi.gen[di][j]} ')
                    fileID.write('\n')
                fileID.write('\n')

            if ri_index != len(R_list) - 1:
                fileID.write('Result:\n')
