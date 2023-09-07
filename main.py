import os
from numpy import zeros, uint8
from pandas import read_csv
from os import getcwd, listdir
from random import seed, randint
from copy import copy, deepcopy

seed(0)


def get_instance(name, folder_dir):
    # Getting #Machines (m) & #Taks (n) from file name
    m = int(name[-6]) * 10
    n = int(name[:-6]) * 100

    # Get DF of instance
    data = read_csv(f'{folder_dir}/{name}', sep="\t", skiprows=[0, 1], header=None)
    # Drop columns of number of machines & NaN's
    data.drop(data.columns[[0] + [i for i in range(1, (m * 2) + 2, 2)]], inplace=True, axis=1)
    # Converting data to matrix
    instance = data.to_numpy(dtype=uint8)
    return m, n, instance


def new_solution(m, n):
    solution = [[] for i in range(m)]

    for i in range(n):
        random_machine = randint(0, m - 1)
        solution[random_machine].append(i)

    return solution


def get_cvalue(m_i, n_j, data):
    return data[n_j][m_i]


def generate_cvalues(m, solution, data):
    c_values = [0 for i in range(m)]

    for i, machine in enumerate(solution):
        for j in machine:
            c_values[i] = c_values[i] + get_cvalue(i, j, data)
    return c_values


def get_cmax(c_values):
    max_val = max(c_values)
    return max_val, c_values.index(max_val)


def count_similars(values, value):
    count = 0
    for v in values:
        if v == value:
            count += 1
    return count - 1


def fitness(c_values):
    part_integer, index = get_cmax(c_values)
    part_decimal = count_similars(c_values, part_integer) / 10
    return part_integer + part_decimal, index


def neighborhood_mapping(solution, m, i_max, c_max, data, c):
    # best_neighbour = []
    # best_i_max = 0
    # best_c_max = c_max
    # best_c = []
    best_neighbour = {'solution': [], 'best_i_max': 0, 'best_c_max': c_max, 'best_c': []}
    neighbor = deepcopy(solution)
    for y, task in enumerate(neighbor[i_max][:]):
        neighbor[i_max].remove(task)
        for m_i in range(m):
            if m_i != i_max:
                neighbor[m_i] = neighbor[m_i] + [task]
                # print('New Neighbor', neighbor)
                new_cs = generate_cvalues(m, neighbor, data)
                new_c_max, new_i_max = fitness(new_cs)
                if new_c_max < best_neighbour['best_c_max']:
                    best_neighbour['solution'] = deepcopy(neighbor)
                    best_neighbour['best_c'] = new_cs[:]
                    best_neighbour['best_c_max'] = new_c_max
                    best_neighbour['best_i_max'] = new_i_max

                neighbor[m_i] = solution[m_i]

        neighbor[i_max].insert(y, task)
    if best_neighbour['best_c_max'] < c_max:
        return False, best_neighbour['solution'], best_neighbour['best_c'], best_neighbour['best_c_max'], best_neighbour['best_i_max']
    else:
        return True, solution, c, c_max, i_max


if __name__ == '__main__':
    # Define max iterarion
    max_iteration = 5
    # Testing creation of new solution
    folder_path = 'Instances/Conjunto'
    # Get all instances in folder
    all_instances = listdir('Instances/Conjunto')
    # Drop 'list.txt'
    all_instances.remove('list.txt')
    # For testing, only use instance 111.txt [5]
    for instance in all_instances:
        machines, tasks, data = get_instance(instance, folder_path)

        with open(f'Results/{instance[:-4]}_R.txt', 'w') as print_results:
            # Start the iteration process
            for iteration in range(max_iteration):
                print(instance)
                # Create a solution
                current_solution = new_solution(machines, tasks)
                print_results.write(f'Iteration: {iteration}\n')
                print_results.write(f'Initial solution: {current_solution}\n')
                # Generate all de C values of solution
                c = generate_cvalues(machines, current_solution, data)
                # Get the Cmax and it indexes
                cmax, i_max = fitness(c)
                # Set the stuck indicator to False
                stuck = False
                while not stuck:
                    stuck, current_solution, c, cmax, i_max = neighborhood_mapping(current_solution, machines, i_max, cmax,
                                                                                   data, c)
                print('Iteration: ', iteration,'Solution:', current_solution, 'Cs:', c, 'Cmax:', cmax, 'C_i:', i_max)

                print_results.write(f'Final solution: {current_solution}\n')
                print_results.write(f"Cmax: {cmax} - C's: {c} - C_i: {i_max}\n")
                print_results.write('\t--------------\t\n')


    # neighbor = deepcopy(a)
    # for y, task in enumerate(neighbor[i_max][:]):
    #     print('Task', task, 'index', y)
    #     neighbor[i_max].remove(task)
    #     for m in range(machines):
    #         if m != i_max:
    #             neighbor[m] = neighbor[m] + [task]
    #             print('New Neighbor', neighbor)
    #             neighbor[m] = a[m]
    #
    #     neighbor[i_max].insert(y, task)
    #     print('Original', a)
    #     print(neighbor)

    # Testing reading and getting instance information
    # # Creating Matrix of data
    # folder_path = 'Instances/Conjunto'
    # # Get all instances in folder
    # all_instances = listdir('Instances/Conjunto')
    #
    # # Drop 'list.txt'
    # all_instances.remove('list.txt')
    # for name_instance in all_instances:
    #     # Get #Machines & #Taks and data info from function.
    #     machines, tasks, data = get_instance(name_instance, folder_path)
    #     print(name_instance)
    #     print(machines, tasks)
    #     print(data)

# Tests ---------------------------------------------
# Know all the file names on dir
# all_instances = listdir('Instances/Conjunto')
# print(all_instances)

#
# # Getting #Machines & #Taks from file name
# m = int(all_instances[0][-5]) * 10
# n = int(all_instances[0][:-6]) * 100
# print(m, n)
#
# print(all_instances[0])
#
# data = read_csv(f'Instances/Conjunto/{all_instances[0]}', sep="\t", skiprows=[0, 1], header=None)
# print(data)
# data.drop(data.columns[[0] + [i for i in range(1, (m * 2) + 2, 2)]], inplace=True, axis=1)
# print(data)
# print(data.to_numpy(dtype=uint8))
