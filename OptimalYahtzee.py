import torch
import torch.nn as nn
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from DeepQNet import DQNAgent
from YahtzeeEnv import YahtzeeEnv


def train_agent(num_episodes=500, print_interval=10, heuristic_start=None, load_filepath=None, save_filepath=None, lr=1e-3, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.010, epsilon_decay=0.995,
                buffer_size=10000, batch_size=32, target_update=100, rng=None,
                alpha=0.6, beta_start=0.4, beta_increment=1e-5):
    """
    Train the DQN agent on YahtzeeEnv for a specified number of episodes.
    Optionally load an existing agent's parameters from 'load_filepath'
    and/or save the agent after training to 'save_filepath'.

    In addition to training, this function now collects the loss (last computed loss in the episode)
    and the average reward (computed over every print_interval episodes) into lists.

    Args:
        num_episodes (int): Number of episodes (full games) to train.
        print_interval (int): Print progress every this many episodes.
        load_filepath (str or None): Path to a saved agent checkpoint.
                                     If provided, loads that agent first.
        save_filepath (str or None): Path to save the trained agent after training.
        (Other hyperparameters omitted for brevity)

    Returns:
        agent: The trained (or further trained) DQN agent.
        loss_history: List of loss values recorded at each print interval.
        avg_reward_history: List of average rewards (over print_interval episodes) recorded at each print interval.
    """
    env = YahtzeeEnv()
    state_dim = env.get_state().shape[0]
    action_dim = 44

    if load_filepath is not None:
        print(f"Loading agent from {load_filepath}...")
        agent = load_agent(load_filepath, state_dim, action_dim)
        agent.epsilon = epsilon_start
    else:
        agent = DQNAgent(state_dim, action_dim, lr=lr, gamma=gamma,
                         epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                         buffer_size=buffer_size, batch_size=batch_size, target_update=target_update, rng=rng,
                         alpha=alpha, beta_start=beta_start, beta_increment=beta_increment)

    if heuristic_start is not None and load_filepath is None:
        initialize_q_values(agent, env)
        play_episode(agent, "initialmodel.md")
        test_policy(num_test_episodes=100)
        print("Initialization done. Training starts...")
        test_agent(agent, num_test_episodes=10)
        save_agent(agent, "heuristic_qvalue.pth")

    total_steps = 0
    episode_rewards = []
    loss_history = []
    avg_reward_history = []
    last_loss = 0.0  # to hold the most recent loss value

    for episode in range(num_episodes):
        env._reset()
        state = env.get_state()
        episode_reward = 0.0
        done = False

        while not done:
            valid_actions = env.get_valid_action()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            next_valid_actions = env.get_valid_action() if not done else []

            agent.push_memory((state, action, reward, next_state, done, next_valid_actions))
            loss = agent.optimize_model()
            if loss is not None:
                last_loss = loss  # update the last computed loss

            state = next_state
            total_steps += 1

            if total_steps % agent.target_update == 0:
                agent.update_target()

        episode_rewards.append(episode_reward)

        try:
            if (agent.epsilon > agent.epsilon_end and (episode % (num_episodes // 600) == 0)) or num_episodes < 600:
                agent.epsilon *= agent.epsilon_decay
        except ZeroDivisionError:
            pass

        if (episode + 1) % print_interval == 0:
            recent_rewards = episode_rewards[-print_interval:]
            avg_reward = np.mean(recent_rewards)
            avg_reward_history.append(avg_reward)
            loss_history.append(last_loss)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor).squeeze(0).cpu().numpy()
                avg_q_value = np.mean(q_values)

            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, Avg Q-value: {avg_q_value:.2f}, "
                  f"Loss: {last_loss:.6f}, Score: {env.scorecard[0:6]}|{env.scorecard[6:]}, "
                  f"Total Score: {np.sum(env.scorecard)}, Epsilon: {agent.epsilon:.3f}")

    if save_filepath is not None:
        save_agent(agent, save_filepath)

    return agent, loss_history, avg_reward_history

def save_plot(loss_data, avg_reward_data, save_filepath):
    """
    Generate and save separate plots for loss and average reward trends over training.
    Also, save the loss and average reward data into a file. The file name is based on the
    same base name as the saved .pth file. If saving as a .mat file is possible, then use .mat;
    otherwise, save as a .csv file.

    The function creates two plots:
        - Loss Trend over Training
        - Average Reward Trend over Training

    Args:
        loss_data (list): List of loss values recorded at each print interval.
        avg_reward_data (list): List of average rewards recorded at each print interval.
        save_filepath (str): The file path used to save the agent (.pth file). The plot and data files
                             will use the same base name.
    """
    base_name = os.path.splitext(save_filepath)[0]

    # Plot for Loss Data
    plt.figure(figsize=(8, 5))
    plt.plot(loss_data, 'b-', label="Loss")
    plt.xlabel('Print Interval Index')
    plt.ylabel('Loss')
    plt.title('Loss Trend over Training')
    plt.legend()
    plt.grid(True)
    loss_plot_file = base_name + '_loss.png'
    plt.savefig(loss_plot_file)
    plt.close()
    print(f"Saved loss plot to {loss_plot_file}.")

    # Plot for Average Reward Data
    plt.figure(figsize=(8, 5))
    plt.plot(avg_reward_data, 'r-', label="Average Reward")
    plt.xlabel('Print Interval Index')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Trend over Training')
    plt.legend()
    plt.grid(True)
    reward_plot_file = base_name + '_reward.png'
    plt.savefig(reward_plot_file)
    plt.close()
    print(f"Saved average reward plot to {reward_plot_file}.")

    # Save data in a .mat file if possible, else in a .csv file.
    try:
        import scipy.io
        data = {
            'loss': loss_data,
            'avg_reward': avg_reward_data,
        }
        data_file = base_name + '.mat'
        scipy.io.savemat(data_file, data)
        print(f"Saved plot data to {data_file} as a .mat file.")
    except ImportError:
        import csv
        data_file = base_name + '.csv'
        with open(data_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['loss', 'avg_reward'])
            for l, r in zip(loss_data, avg_reward_data):
                writer.writerow([l, r])
        print(f"Saved plot data to {data_file} as a .csv file.")


def initialize_q_values(agent, env):
    """
    Heuristic Policy를 이용해 초기 Q-values를 설정
    """
    for _ in range(50000):  # 충분한 상태를 샘플링하여 초기 Q값 학습
        env._reset()
        state = env.get_state()
        done = False
        step = 0

        while not done:
            action = env.heuristic_policy()
            next_state, reward, done, _ = env.step(action)

            # Heuristic 기반 Value Function으로 Q-value 초기화
            target_q_value = reward + agent.gamma * torch.max(agent.policy_net(torch.FloatTensor(next_state)))
            agent.policy_net(torch.FloatTensor(state))[action] = target_q_value

            state = next_state
        step += 1
        if step% 1000 == 0:
            print(f"Step {step} done.")

def test_agent(agent, num_test_episodes=20):
    """
    Test a trained DQN agent (no/low exploration) on YahtzeeEnv.
    
    Args:
        agent (DQNAgent): A trained DQNAgent (with .policy_net on agent.device).
        num_test_episodes (int): Number of episodes to test (full Yahtzee games).
        
    Returns:
        float: The average reward over the test episodes.
    """
    # Temporarily store the old epsilon, then set epsilon to 0 for pure exploitation.
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during testing

    # We'll move input data to the same device as agent.policy_net
    device = agent.device

    env = YahtzeeEnv()
    rewards = []

    for episode in range(num_test_episodes):
        env._reset()
        state = env.get_state()
        episode_reward = 0.0
        done = False
        
        while not done:
            valid_actions = env.get_valid_action()
            
            # Inference on GPU (or CPU if CUDA not available)
            with torch.no_grad():
                # Move the state to the same device as the model
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Forward pass on the policy network
                q_values_tensor = agent.policy_net(state_tensor).squeeze(0)  
                
                # Move back to CPU for NumPy-based masking
                q_values = q_values_tensor.cpu().numpy()
            
            # Create a masked Q-value array that sets invalid actions to -∞
            masked_q_values = np.full(agent.action_dim, -np.inf)
            for a in valid_actions:
                masked_q_values[a] = q_values[a]
            
            # Choose the best action among valid actions (greedy)
            best_action = int(np.argmax(masked_q_values))
            
            # Step in the environment (on CPU)
            next_state, reward, done, _ = env.step(best_action)
            episode_reward += reward
            
            # Move on to the next state
            state = next_state

        rewards.append(episode_reward)

    # Restore agent’s original epsilon
    agent.epsilon = old_epsilon

    # Compute average reward
    avg_reward = np.mean(rewards)
    print(f"Tested on {num_test_episodes} episodes. Avg reward = {avg_reward:.2f}")
    return avg_reward

def test_policy(num_test_episodes=50):
    rewards = []
    for i in range(num_test_episodes):
        reward = play_episode_with_policy()
        rewards.append(reward)
    avg_reward = np.mean(rewards)
    print(f"Average reward for this policy: {avg_reward:.3f}")
    
def save_agent(agent, filepath="trained_agent.pth"):
    """
    Save the trained DQN agent's policy network and parameters.

    Args:
        agent (DQNAgent): The trained agent to save.
        filepath (str): Path to save the model.
    """
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, filepath)
    print(f"Agent saved to {filepath}")

def save_model_info(model, optimizer, loss_fn, filename="modelinfo.md"):
    with open(filename, "w") as file:
        file.write("# Model Information\n\n")
        
        # 모델의 layer 구조
        file.write("## Model Architecture\n")
        file.write("```python\n")
        file.write(str(model) + "\n")
        file.write("```\n\n")
        
        # Optimizer 정보
        file.write("## Optimizer\n")
        file.write(f"Optimizer: {optimizer.__class__.__name__}\n")
        file.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        file.write(f"Parameters: {optimizer.param_groups[0]}\n\n")
        
        # Loss function 정보
        file.write("## Loss Function\n")
        file.write(f"Loss Function: {loss_fn.__class__.__name__}\n\n")
        
        # Activation functions 정보 (모델에서 사용하는 activation function 추출)
        file.write("## Activation Functions\n")
        activation_functions = []
        for layer in model.children():
            if isinstance(layer, nn.ReLU):
                activation_functions.append("ReLU")
            elif isinstance(layer, nn.Sigmoid):
                activation_functions.append("Sigmoid")
            elif isinstance(layer, nn.Tanh):
                activation_functions.append("Tanh")
        
        if activation_functions:
            file.write(f"Used activation functions: {', '.join(activation_functions)}\n")
        else:
            file.write("No activation function found\n")

def load_agent(filepath, state_dim, action_dim):
    """
    Load a trained DQN agent from a file.

    Args:
        filepath (str): Path to the saved model checkpoint.
        state_dim (int): State dimension (should match training).
        action_dim (int): Action dimension (should match training).

    Returns:
        DQNAgent: The loaded agent.
    """
    # Create a new agent instance with the same architecture
    agent = DQNAgent(state_dim, action_dim)

    # Load the checkpoint file
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))  # Use CPU for portability
    
    # Restore model parameters
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    
    print(f"Agent loaded from {filepath}")
    return agent

def find_latest_trial(num_episodes):
    """
    Find the largest trial number among files in the current directory
    matching the pattern: trained_agent_{trial}_{num_episodes}.pth

    Returns:
        int: the largest trial found, or -1 if no matching file exists.
    """
    pattern = re.compile(r"^trained_agent_(\d+)_(\d+)\.pth$")
    max_trial = -1

    # Check every file in the current working directory
    for fname in os.listdir('.'):
        match = pattern.match(fname)
        if match:
            found_trial = int(match.group(1))
            found_episodes = int(match.group(2))
            if found_episodes == num_episodes and found_trial > max_trial:
                max_trial = found_trial

    return max_trial

def play_episode(agent, md_filename="yahtzee_playthrough.md"):
    """
    Plays one episode of Yahtzee with a trained agent, writing each step's info to a Markdown file.
    """
    env = YahtzeeEnv()
    env._reset()
    done = False
    steps = 0
    cumulative_reward = 0
    category = {
    0: "initiate roll",  # 처음 굴리기

    # 주사위를 다시 굴리는 행동
    1:  "reroll 00001",  2:  "reroll 00010",  3:  "reroll 00011",  4:  "reroll 00100",
    5:  "reroll 00101",  6:  "reroll 00110",  7:  "reroll 00111",  8:  "reroll 01000",
    9:  "reroll 01001", 10:  "reroll 01010", 11: "reroll 01011", 12: "reroll 01100",
    13: "reroll 01101", 14: "reroll 01110", 15: "reroll 01111", 16: "reroll 10000",
    17: "reroll 10001", 18: "reroll 10010", 19: "reroll 10011", 20: "reroll 10100",
    21: "reroll 10101", 22: "reroll 10110", 23: "reroll 10111", 24: "reroll 11000",
    25: "reroll 11001", 26: "reroll 11010", 27: "reroll 11011", 28: "reroll 11100",
    29: "reroll 11101", 30: "reroll 11110", 31: "reroll 11111",

    # 점수를 기록하는 행동
    32: "score : ones", 33: "score : twos", 34: "score : threes", 35: "score : fours",
    36: "score : fives", 37: "score : sixes", 38: "score : choices", 39: "score : four of a kind",
    40: "score : full house", 41: "score : small straight", 42: "score : large straight",
    43: "score : yahtzee",
}


    # We prepare lines of markdown
    md_lines = []
    md_lines.append("# Yahtzee Episode Playthrough\n")
    md_lines.append("**Environment:** Yahtzee\n")
    md_lines.append("**Agent:** Trained DQN (placeholder)\n")
    md_lines.append("---\n")

    md_lines.append("## Step-by-Step Decisions\n")
    md_lines.append("| Step | Dice (One-Hot) | Rerolls | Turn | Valid Actions | Chosen Action | Reward | Cumulative Reward | Done? |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    # Start the loop
    state = env.get_state()
    while not done and steps < 100:  # 12 turns is typical, 100 is a safe upper bound
        valid_actions = env.get_valid_action()
        action = agent.select_action(state, valid_actions)
        agent.epsilon = 0
        next_state, reward, done, info = env.step(action)

        steps += 1
        cumulative_reward += reward

        # Convert dice to a more readable list of faces
        dice_desc = []
        for i in range(5):
            face_idx = np.argmax(env.dice[i])  # which face is "1"
            dice_desc.append(str(face_idx+1))
        dice_str = ", ".join(dice_desc)

        # Summarize step in table row
        line = f"| {steps} | **{dice_str}** | {next_state[30]} | {next_state[31]} | `{next_state[32:44].astype(int)}` | **{category[action]}** | {reward:.2f} | {cumulative_reward:.2f} | {done} |"
        md_lines.append(line)

        state = next_state

    # Summarize final score
    final_score = np.sum(env.scorecard)
    bonus_desc = f"(Bonus Active)" if env.bonus else ""
    md_lines.append("\n---\n")
    md_lines.append(f"**Episode finished** after **{steps}** steps.\n\n")
    md_lines.append(f"**Final Scorecard** = {env.scorecard}  \n")
    md_lines.append(f"**Sum of Scorecard** = {final_score} {bonus_desc}\n")
    md_lines.append(f"**Cumulative Reward** = {cumulative_reward:.2f}\n")

    # Write to Markdown file
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Playthrough complete. Markdown log written to {md_filename}.")

def play_episode_with_policy(md_filename="yahtzee_playthrough_with_policy.md"):
    """play episode with policy function

    Args:
        policy (func, output:int ): policy function whose output is action for given state
    """
    env = YahtzeeEnv()
    env._reset()
    done = False
    steps = 0
    cumulative_reward = 0
    category = {
    0: "initiate roll",  # 처음 굴리기

    # 주사위를 다시 굴리는 행동
    1:  "reroll 00001",  2:  "reroll 00010",  3:  "reroll 00011",  4:  "reroll 00100",
    5:  "reroll 00101",  6:  "reroll 00110",  7:  "reroll 00111",  8:  "reroll 01000",
    9:  "reroll 01001", 10:  "reroll 01010", 11: "reroll 01011", 12: "reroll 01100",
    13: "reroll 01101", 14: "reroll 01110", 15: "reroll 01111", 16: "reroll 10000",
    17: "reroll 10001", 18: "reroll 10010", 19: "reroll 10011", 20: "reroll 10100",
    21: "reroll 10101", 22: "reroll 10110", 23: "reroll 10111", 24: "reroll 11000",
    25: "reroll 11001", 26: "reroll 11010", 27: "reroll 11011", 28: "reroll 11100",
    29: "reroll 11101", 30: "reroll 11110", 31: "reroll 11111",

    # 점수를 기록하는 행동
    32: "score : ones", 33: "score : twos", 34: "score : threes", 35: "score : fours",
    36: "score : fives", 37: "score : sixes", 38: "score : choices", 39: "score : four of a kind",
    40: "score : full house", 41: "score : small straight", 42: "score : large straight",
    43: "score : yahtzee",
}


    # We prepare lines of markdown
    md_lines = []
    md_lines.append("# Yahtzee Episode Playthrough\n")
    md_lines.append("**Environment:** Yahtzee\n")
    md_lines.append("**Agent:** Trained DQN (placeholder)\n")
    md_lines.append("---\n")

    md_lines.append("## Step-by-Step Decisions\n")
    md_lines.append("| Step | Dice (One-Hot) | Rerolls | Turn | Valid Actions | Chosen Action | Reward | Cumulative Reward | Done? |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    # Start the loop
    state = env.get_state()
    while not done and steps < 100:  # 12 turns is typical, 100 is a safe upper bound
        action = env.heuristic_policy()
        next_state, reward, done, info = env.step(action)
        steps += 1
        cumulative_reward += reward

        # Convert dice to a more readable list of faces
        dice_desc = []
        for i in range(5):
            face_idx = np.argmax(env.dice[i])  # which face is "1"
            dice_desc.append(str(face_idx+1))
        dice_str = ", ".join(dice_desc)

        # Summarize step in table row
        line = f"| {steps} | **{dice_str}** | {next_state[30]} | {next_state[31]} | `{next_state[32:44].astype(int)}` | **{category[action]}** | {reward:.2f} | {cumulative_reward:.2f} | {done} |"
        md_lines.append(line)

        state = next_state

    # Summarize final score
    final_score = np.sum(env.scorecard)
    bonus_desc = f"(Bonus Active)" if env.bonus else ""
    md_lines.append("\n---\n")
    md_lines.append(f"**Episode finished** after **{steps}** steps.\n\n")
    md_lines.append(f"**Final Scorecard** = {env.scorecard}  \n")
    md_lines.append(f"**Sum of Scorecard** = {final_score} {bonus_desc}\n")
    md_lines.append(f"**Cumulative Reward** = {cumulative_reward:.2f}\n")

    # Write to Markdown file
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Playthrough complete. Markdown log written to {md_filename}.")
    return final_score




if __name__ == "__main__":

    # Just a debug check: prints True if GPU is available
    print("CUDA available?", torch.cuda.is_available())
    #print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.device("cpu")

    # Number of training episodes for this run
    num_episodes = 500

    # 1) Look for an existing trial file in the current directory
    trial = find_latest_trial(num_episodes)
    if trial >= 0:
        # Found a file trained_agent_{trial}_{num_episodes}.pth
        load_filepath = f"trained_agent_{trial}_{num_episodes}.pth"
        print(f"Loading from file: {load_filepath}")
    else:
        load_filepath = None  # No prior file found
        print("No prior training file found; starting from scratch.")

    # 2) Define where to save next trial
    #    e.g. if trial = 2, we save next as trained_agent_3_{num_episodes}.pth
    save_filepath = f"trained_agent_{trial+1}_{num_episodes}.pth"

    # 3) Train (or continue training) the agent and save the model to save_filepath
    rng = np.random.default_rng()
    trained_agent, loss_data, avg_reward_data = train_agent(
        num_episodes=num_episodes,
        print_interval=50,
        heuristic_start=None,
        load_filepath=load_filepath,      # Might be None if not found
        save_filepath=save_filepath, 
        lr=1e-4, 
        gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.080, epsilon_decay=0.996,
        buffer_size=100000, batch_size=256, target_update=250, rng=rng
    )

    # 4) Test the agent
    test_score = test_agent(trained_agent, num_test_episodes=500)
    print(f"Final average test reward: {test_score:.2f}")
    play_episode(trained_agent)

    # 5) save the model info in modelinfo.md
    save_model_info(trained_agent.policy_net.net, trained_agent.optimizer, nn.MSELoss())
    
    #6) save the plot
    save_plot(loss_data, avg_reward_data, save_filepath)
    print("Training complete.")
    
    