import os
from numpy import zeros, uint8
from pandas import read_csv
from os import getcwd, listdir
from random import seed, randint
from copy import copy, deepcopy
from time import time
from statistics import mean, stdev, median

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
        return False, best_neighbour['solution'], best_neighbour['best_c'], best_neighbour['best_c_max'], \
            best_neighbour['best_i_max']
    else:
        return True, solution, c, c_max, i_max


def neighbour_evaluation_sw10(cmax, cs, i_max, i_target, task, data):
    update = False
    neighbour_c = [c for c in cs]
    neighbour_c[i_max] = cs[i_max] - get_cvalue(i_max, task, data)
    neighbour_c[i_target] = cs[i_target] + get_cvalue(i_target, task, data)
    new_c_max, _ = fitness(neighbour_c)
    if new_c_max < cmax:
        update = True
    return update, new_c_max


def neighbour_evaluation_sw20(cmax, cs, i_max, target_1, target_2, task_1, task_2, data):
    update = False
    neighbour_c = [c for c in cs]
    neighbour_c[i_max] = neighbour_c[i_max] - get_cvalue(i_max, task_1, data)
    neighbour_c[i_max] = neighbour_c[i_max] - get_cvalue(i_max, task_2, data)
    neighbour_c[target_1] = neighbour_c[target_1] + get_cvalue(target_1, task_1, data)
    neighbour_c[target_2] = neighbour_c[target_2] + get_cvalue(target_2, task_2, data)
    new_c_max, _ = fitness(neighbour_c)
    if new_c_max < cmax:
        update = True
    return update, new_c_max


def swap_1_0(solution, origin, target, task):
    swaped_solution = [m for m in solution]
    swaped_solution[origin].remove(task)
    swaped_solution[target] = swaped_solution[target] + [task]
    return swaped_solution


def swap_2_0(solution, origin, target_1, target_2, task_1, task_2):
    swaped_solution = [m for m in solution]
    swaped_solution[origin].remove(task_1)
    swaped_solution[target_1] = swaped_solution[target_1] + [task_1]
    swaped_solution[origin].remove(task_2)
    swaped_solution[target_2] = swaped_solution[target_2] + [task_2]
    return swaped_solution


def neighborhood_mapping_sw10(solution, m, i_max, c_max, data, c, moves, good_neighbours):
    # Create best target machine, task and cmax variables
    best_target = 0
    best_task = 0
    best_cmax = c_max
    # Set found_solution to false. Only change when/if a better neighbour is found.
    found_solution = False

    # Iterate over the task of the machine with Cmax
    for task in solution[i_max]:
        # Iterate over all machines
        for m_i in range(m):
            # Skip the origin machine to avoid repetitions
            if m_i != i_max:
                # Evaluate the neighbour
                update, new_cmax = neighbour_evaluation_sw10(best_cmax, c, i_max, m_i, task, data)
                print('Cmax: ', c_max, 'Best: ', best_cmax, 'Good N', good_neighbours, 'Moves: ', moves)
                # if the neighbour is better, update the best values
                if update:
                    # Update the counter for improvements
                    good_neighbours += 1
                    # Best target is the current machine
                    best_target = m_i
                    # Best task is the current task
                    best_task = task
                    # Best Cmax is the current Cmax
                    best_cmax = new_cmax
                    # A better solution was found
                    found_solution = True
    # If a better solution was found, create the neighbour.
    if found_solution:
        # Update the counter of neighbour created
        moves += 1
        # Build the neighbour
        solution = swap_1_0(solution, i_max, best_target, best_task)
        # Get the C values
        c = generate_cvalues(m, solution, data)
        # Get the Cmax and its index
        c_max, i_max = fitness(c)
    # Return results
    return found_solution, solution, c, c_max, i_max, moves, good_neighbours


def neighborhood_mapping_sw20(solution, m, i_max, c_max, data, c, moves, good_neighbours):
    # Create best target machines, tasks and cmax variables
    best_target = (0, 0)
    best_task = (0, 0)
    best_cmax = c_max
    # Set found_solution to false. Only change when/if a better neighbour is found.
    found_solution = False

    # Iterate over the task of the machine with Cmax for the first element
    for x in range(len(solution[i_max])):
        for y in range(x + 1, len(solution[i_max])):
            # Iterate over all machines for the first target
            for m_x in range(m):
                # Skip the origin machine to avoid repetitions
                if m_x != i_max:
                    # Iterate over all machines for the second target
                    for m_y in range(m):
                        # Skip the origin machine to avoid repetitions
                        if m_y != i_max:
                            # Evaluate the neighbour
                            update, new_cmax = neighbour_evaluation_sw20(best_cmax, c, i_max, m_x, m_y, solution[i_max][x], solution[i_max][y], data)
                            print('Cmax: ', c_max, 'Best: ', best_cmax, 'Good N', good_neighbours, 'Moves: ', moves)
                            # if the neighbour is better, update the best values
                            if update:
                                # Update the counter for improvements
                                good_neighbours += 1
                                # Best targets are the current machines
                                best_target = (m_x, m_y)
                                # Best tasks are the current task
                                best_task = (solution[i_max][x], solution[i_max][y])
                                # Best Cmax is the current Cmax
                                best_cmax = new_cmax
                                # A better solution was found
                                found_solution = True
    # If a better solution was found, create the neighbour.
    if found_solution:
        print('Found')
        # Update the counter of neighbour created
        moves += 1
        # Build the neighbour
        solution = swap_2_0(solution, i_max, best_target[0], best_target[1], best_task[0], best_task[1])
        # Get the C values
        c = generate_cvalues(m, solution, data)
        # Get the Cmax and its index
        c_max, i_max = fitness(c)
    # Return results
    return found_solution, solution, c, c_max, i_max, moves, good_neighbours


if __name__ == '__main__':
    # Define max iterarion
    max_iteration = 3
    history_cmax_iterations = [0 for iteration_x in range(max_iteration)]
    history_time_iterations = [0 for iteration_x in range(max_iteration)]
    # Testing creation of new solution
    folder_path = 'Instances/Conjunto'
    # Get all instances in folder
    all_instances = listdir('Instances/Conjunto')
    # Drop 'list.txt'
    all_instances.remove('list.txt')
    # For testing, only use instance 111.txt [5]
    for instance in [all_instances[5]]:
        machines, tasks, data = get_instance(instance, folder_path)

        with open(f'Results/{instance[:-4]}_R-T.txt', 'w') as print_results:
            # Start the iteration process
            for iteration in range(max_iteration):
                print(instance)
                total_neighbours = 0
                total_movements = 0
                history_cmax = []
                start_time = time()
                # Create a solution
                current_solution = new_solution(machines, tasks)
                print_results.write(f'Iteration: {iteration}\n')
                print_results.write(f'Initial solution: {current_solution}\n')
                # Generate all de C values of solution
                c = generate_cvalues(machines, current_solution, data)
                # Get the Cmax and it indexes
                cmax, i_max = fitness(c)
                # Set the stuck indicator to False
                not_stuck = True
                while not_stuck:
                    not_stuck, current_solution, c, cmax, i_max, total_movements, total_neighbours = neighborhood_mapping_sw20(
                        current_solution, machines, i_max,
                        cmax,
                        data, c, total_movements, total_neighbours)
                    history_cmax.append(cmax)
                dt = time() - start_time
                print('Iteration: ', iteration, 'Solution:', current_solution, 'Cs:', c, 'Cmax:', cmax, 'C_i:', i_max,
                      'Time:', dt)
                print_results.write(f'Final solution: {current_solution}\n')
                print_results.write(f"Cmax: {cmax} - C's: {c} - C_i: {i_max} - Time: {dt}\n")
                print_results.write(f"Total movements: {total_movements} - Total neighbours: {total_neighbours} \n")
                print_results.write(f"Cmaxs: {history_cmax}\n")
                print_results.write('\t--------------\t\n')
                history_cmax_iterations[iteration] = cmax
                history_time_iterations[iteration] = dt
            print_results.write('\t--------------------------------------------------------\t\n')
            print_results.write('\t--------------------------------------------------------\t\n')
            print_results.write(f'Cmax: {history_cmax_iterations}\n')
            mean_value = mean(history_cmax_iterations)
            stdev_value = stdev(history_cmax_iterations)
            median_value = median(history_cmax_iterations)
            max_value = max(history_cmax_iterations)
            min_value = min(history_cmax_iterations)
            print_results.write(f'Cmax mean: {mean_value} - Cmax stdev: {stdev_value}\n')
            print_results.write(f'Cmax MAX: {max_value, history_cmax_iterations.index(max_value)} ')
            print_results.write(f'Cmax median: {median_value, history_cmax_iterations.index(median_value)} ')
            print_results.write(f'Cmax min: {min_value, history_cmax_iterations.index(min_value)}\n\n')

            print_results.write(f'Times: {history_time_iterations}\n')
            mean_value = mean(history_time_iterations)
            stdev_value = stdev(history_time_iterations)
            median_value = median(history_time_iterations)
            max_value = max(history_time_iterations)
            min_value = min(history_time_iterations)
            print_results.write(f'Time mean: {mean_value} - Time stdev: {stdev_value}\n')
            print_results.write(f'Time MAX: {max_value, history_time_iterations.index(max_value)} ')
            print_results.write(f'Time median: {median_value, history_time_iterations.index(median_value)}')
            print_results.write(f'Time min: {min_value, history_time_iterations.index(min_value)}')

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
