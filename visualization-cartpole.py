# This is a single code that helps in visualizing all the losses / rewards and other metrics for 
# all the algorithms. 
import numpy as np
import matplotlib.pyplot as plt


def load_npy_file(file_name):
    # This function loads the npy file and returns the npy file.
    loaded_npy_file = np.load(file_name)
    return loaded_npy_file

def visualize_rewards(folder_path):
    runs_reward_list = []
    max_length_of_all_files = 0
    for i in range(10):
        file_name = folder_path + "/cartpole-per_episode_reward-" + str(i) + ".npy"
        loaded_npy_file = load_npy_file(file_name)
        # Check the size of the loaded_npy_file
        # Extract the second column from the loaded_npy_file
        # Append this to the runs_reward_list
        second_col = loaded_npy_file[:, 1]
        if second_col.shape[0] > max_length_of_all_files:
            print("Max Length of all files: ", max_length_of_all_files)
            max_length_of_all_files = second_col.shape[0]
        # Now check if the size of the second_col is less than 3000
        # If yes, then pad the array with 500's.
        if second_col.shape[0] < 3000:
            second_col = np.pad(second_col, (0, 3000 - second_col.shape[0]), 'constant', constant_values=(500))
        runs_reward_list.append(second_col.tolist())
        # If the size is less than 3000 in dim 0 , then pad the array with 500's.
    # Now we have the runs_reward_list.
    # Now we can plot the graph where the x-axis is the episode number and the y-axis is the reward.
    # We can plot the mean and the standard deviation.
    # Iterate through the runs_reward_list, trim only to the max_length_of_all_files.
    for i in range(len(runs_reward_list)):
        if len(runs_reward_list[i]) > max_length_of_all_files:
            runs_reward_list[i] = runs_reward_list[i][:max_length_of_all_files]

    # Calculate the mean and the standard deviation.

    means = np.mean(runs_reward_list, axis=0)
    std_devs = np.std(runs_reward_list, axis=0)

    # Plotting mean and standard deviation
    fig, ax = plt.subplots()

    # Plot mean
    ax.plot(means, label='Mean')

    # Plot standard deviation as a shaded region
    ax.fill_between(range(max_length_of_all_files), means - std_devs, means + std_devs, alpha=0.2, label='Standard Deviation')

    # Customize the plot
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_title('Mean and S.D of rewards for 10 runs with best hyper-params')
    ax.legend()

    # Plot the second one i.e cartpole_logs/cartpole-per_episode_reward-{i}.npy on top of this in red line.
    #best_run = load_npy_file("cartpole_logs/cartpole-per_episode_reward-0.npy")
    #best_col = best_run[:, 1][:max_length_of_all_files]
    #fill the second_col if the size is less than 3000
    #if best_col.shape[0] < max_length_of_all_files:
    #    best_col = np.pad(best_col, (0, max_length_of_all_files - best_col.shape[0]), 'constant', constant_values=(500))
    
    # plot the best run on top of this.
    #ax.plot(best_col, color='red', label='Best Run')
    #plt.legend()

    # Show the plot
    plt.show()

def visualize_rewards_second(folder_path):
    runs_reward_list = []
    max_length_of_all_files = 0
    for i in range(10):
        file_name = folder_path + "/cartpole-per_episode_reward-" + str(i) + ".npy"
        loaded_npy_file = load_npy_file(file_name)
        # Check the size of the loaded_npy_file
        # Extract the second column from the loaded_npy_fil
        if loaded_npy_file.shape[0] > max_length_of_all_files and loaded_npy_file.shape[0] != 3000 :
            max_length_of_all_files = loaded_npy_file.shape[0]  
        #print("Loaded NPY File: ", loaded_npy_file)
    for i in range(10):
        file_name = folder_path + "/cartpole-per_episode_reward-" + str(i) + ".npy"
        loaded_npy_file = load_npy_file(file_name)
        # Truncate the file to contain only max_length_of_all_files.
        # extract only the second column.
        second_col = loaded_npy_file[:, 1][:max_length_of_all_files]
        # Now check if the size of the second_col is less than max_length_of_all_files
        # If yes, then pad the array with 500's.
        if second_col.shape[0] < max_length_of_all_files:
            second_col = np.pad(second_col, (0, max_length_of_all_files - second_col.shape[0]), 'constant', constant_values=(500))
        print("Second Col: ", second_col.shape)
        runs_reward_list.append(second_col.tolist())

    means = np.mean(runs_reward_list, axis=0)
    std_devs = np.std(runs_reward_list, axis=0)
    fig, ax = plt.subplots()
    ax.plot(means, label='Mean')
    ax.fill_between(range(max_length_of_all_files), means - std_devs, means + std_devs, alpha=0.2, label='Standard Deviation')

    # Customize the plot
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_title('Mean and S.D of rewards with best-hyperparams')
    ax.legend()


    # The best one is cartpole_logs/cartpole-per_episode_reward-0.npy
    #best_training_run = runs_reward_list[0]
    #ax.plot(best_training_run, color='red', label='Best Run')

    plt.show()


        
        

def plot_step_per_each_episode():
    # This plot contains the graph with the number of steps taken per episode.
    cartpole_step_episode_0 = load_npy_file("cartpole_logs/cartpole-step_per_each_episodes-0.npy")
    # get the second column from the npy file.
    second_col = cartpole_step_episode_0[:, 1]
    # do cumulative sum on the second column.
    cum_sum_second_col = np.cumsum(second_col)
    # plot the graph.
    #plt.plot(second_col, label="Steps taken per episode")
    plt.plot(cum_sum_second_col, label="Cumulative sum of steps")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Successful balanced steps taken by the agent")
    plt.title("Plot indicating learning of the agent over episodes.")
    plt.legend()
    plt.show()
    
def print_best_hyper_params():
    # This function prints the best hyper-parameters.
    cartpole_best_hyper_params = load_npy_file("cartpole_logs/cartpole-best_hyperparameter_dict.npy", allow_pickle=True)
    print(cartpole_best_hyper_params)

if __name__ == "__main__":
    # Lets discuss about how the npy looks like : 
    # cartpole-per_episode_reward-1 : : 
    # This contains - epi_rewards.append((episode, total_reward))
    cartpole_per_episode_reward_1 = load_npy_file("cartpole_logs/cartpole-per_episode_reward-1.npy")
    #print(cartpole_per_episode_reward_1.shape)
    visualize_rewards_second("cartpole_logs")
    #plot_step_per_each_episode()
    # print(best hyper-params)
    #print_best_hyper_params()

