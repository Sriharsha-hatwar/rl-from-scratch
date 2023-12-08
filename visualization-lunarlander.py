import numpy as np
import matplotlib.pyplot as plt

def load_npy_file(file_name):
    # This function loads the npy file and returns the npy file.
    loaded_npy_file = np.load(file_name)
    return loaded_npy_file

def visualize_rewards(folder_path):
    runs_reward_list = []
    for i in range(8):
        file_name = folder_path + "/lunar-per_episode_reward-" + str(i) + ".npy"
        loaded_npy_file = load_npy_file(file_name)
        second_col = loaded_npy_file[:, 1]
        if second_col.shape[0] < 30000:
            second_col = np.pad(second_col, (0, 30000 - second_col.shape[0]), 'constant', constant_values=(230))
        runs_reward_list.append(second_col.tolist())
    means = np.mean(runs_reward_list, axis=0)
    std_devs = np.std(runs_reward_list, axis=0)
    fig, ax = plt.subplots()
    ax.plot(means, label='Mean')
    ax.fill_between(range(30000), means - std_devs, means + std_devs, alpha=0.2, label='Standard Deviation')

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_title('Mean and S.D of rewards for 8 runs')
    ax.legend()

    
    plt.legend()
    plt.show()

def plot_step_per_each_episode():
    # This plot contains the graph with the number of steps taken per episode.
    lunar_step_episode_6 = load_npy_file("lunar_lander_logs/lunar-per_episode_step_count-6.npy")
    # get the second column from the npy file.
    second_col = lunar_step_episode_6[:, 1]
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

if __name__ == "__main__":
    visualize_rewards("lunar_lander_logs")
    #plot_step_per_each_episode()