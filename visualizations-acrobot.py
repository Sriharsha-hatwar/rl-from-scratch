import numpy as np
import matplotlib.pyplot as plt

# 3rd time, 7th time, original one. 

def load_npy_file(file_name):
    # This function loads the npy file and returns the npy file.
    loaded_npy_file = np.load(file_name)
    return loaded_npy_file

def load_mutated_list(file_name):
    # This function loads the npy file and returns the npy file.
    loaded_npy_file = load_npy_file(file_name)
    second_col = loaded_npy_file[:, 1]
    last_value = second_col[-1]
    original_size = second_col.shape[0]
    if second_col.shape[0] < 3000:
        second_col = np.pad(second_col, (0, 3000 - second_col.shape[0]), 'constant', constant_values=(last_value))
    return second_col.tolist(), original_size
    

def visualize_rewards(folder_path):
    runs_reward_list = []

    third_one, third_shape = load_mutated_list(folder_path + "/acrobot-per_episode_reward-3.npy")
    seventh_one, seventh_shape = load_mutated_list(folder_path + "/acrobot-per_episode_reward-7.npy")
    original_one, original_shape = load_mutated_list(folder_path + "/pre-saving-acrobot-epi-rewards.npy")
    #first_one, first_shape = load_mutated_list(folder_path + "/acrobot-per_episode_reward-0.npy")
    #second_one, second_shape = load_mutated_list(folder_path + "/acrobot-per_episode_reward-1.npy")

    # Now we have the runs_reward_list.
    runs_reward_list.append(third_one)
    runs_reward_list.append(seventh_one)
    runs_reward_list.append(original_one)
    #runs_reward_list.append(first_one)
    #runs_reward_list.append(second_one)

    means = np.mean(runs_reward_list, axis=0)[:1500]
    std_devs = np.std(runs_reward_list, axis=0)[:1500]

    fig, ax = plt.subplots()
    ax.plot(means[:1500], label='Mean')
    ax.fill_between(range(1500), means - std_devs, means + std_devs, alpha=0.2, label='Standard Deviation')

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_title('Mean and S.D of rewards for 5 runs with best hyper-params')
    ax.legend()

    # Plot the second one i.e cartpole_logs/cartpole-per_episode_reward-{i}.npy on top of this in red line.
    #best_col = seventh_one
    #fill the second_col if the size is less than 3000
    #best_col = best_col[:seventh_shape]

    #print("Best Col Shape: ", best_col)
    
    # plot the best run on top of this.
    #ax.plot(best_col, color='red', label='Best Run')
    plt.legend()

    # Show the plot
    plt.show()

def plot_step_per_each_episode(folder_path):
    seventh_one, seventh_shape = load_mutated_list(folder_path + "/acrobot-per_episode_reward-7.npy")
    # conver the rewards to steps by multilying by -1. 
    print("Seventh One: ", seventh_one)
    oringal_seventh = seventh_one[:seventh_shape]
    # Now change the sign of the seventh_one.
    steps = np.multiply(oringal_seventh, -1)
    print("Steps: ", steps)
    # Now do a cumulative sum on the steps.
    cum_sum_steps = np.cumsum(steps)
    print("Cumulative Sum Steps: ", cum_sum_steps)

    #second_one, second_shape = load_mutated_list(folder_path + "/acrobot-per_episode_reward-1.npy")
    # conver the rewards to steps by multilying by -1.
    #oringal_second = second_one[:second_shape]
    # Now change the sign of the second_one.
    #steps = np.multiply(oringal_second, -1)
    # Now do a cumulative sum on the steps.
    #cum_second_steps = np.cumsum(steps)[:1000]



    # plot the graph.
    plt.plot(cum_sum_steps, label="Cumulative sum of steps - With learning")
    #plt.plot(cum_second_steps, label="Cumulative sum of steps - No learning", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative steps taken by the agent")
    plt.title("Plot indicating learning of the agent over episodes.")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    visualize_rewards("acrobot_logs")
    #plot_step_per_each_episode("acrobot_logs")


