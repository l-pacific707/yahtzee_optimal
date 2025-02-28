import torch
import torch.nn as nn
import numpy as np
import os
import re
from DeepQNet import DQNet
from DeepQNet import DQNAgent
from YahtzeeEnv import YahtzeeEnv

def train_agent(num_episodes=500, print_interval=10, load_filepath=None, save_filepath=None, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.010, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100, rng=None):
    """
    Train the DQN agent on YahtzeeEnv for a specified number of episodes.
    Optionally load an existing agent's parameters from 'load_filepath'
    and/or save the agent after training to 'save_filepath'.
    
    Args:
        num_episodes (int): Number of episodes (full games) to train.
        print_interval (int): Print progress every this many episodes.
        load_filepath (str or None): Path to a saved agent checkpoint. 
                                     If provided, loads that agent first.
        save_filepath (str or None): Path to save the trained agent after training.
    
    Returns:
        DQNAgent: The trained (or further trained) DQN agent.
    """

    # Initialize environment.
    env = YahtzeeEnv()
    # The state dimension is determined from the environment.
    state_dim = env.get_state().shape[0]
    # As defined in YahtzeeEnv, there are 43 discrete actions (1..43).
    action_dim = 44

    # 1) Either load an existing agent or create a new one.
    if load_filepath is not None:
        # We load from a checkpoint file.
        print(f"Loading agent from {load_filepath}...")
        agent = load_agent(load_filepath, state_dim, action_dim)
        agent.epsilon = epsilon_start
    else:
        # We create a new agent from scratch.
        agent = DQNAgent(state_dim, action_dim, lr=lr, gamma=gamma,
                 epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                 buffer_size=buffer_size, batch_size=batch_size, target_update=target_update, rng=rng)
    
    total_steps = 0  # Counts total steps across episodes (for target_net updates).

    for episode in range(num_episodes):
        # Reset the environment for a new episode (a new Yahtzee game).
        env._reset()
        state = env.get_state()
        
        episode_reward = 0.0
        done = False
        
        while not done:
            # Retrieve valid actions for the current state.
            valid_actions = env.get_valid_action()
            
            # Agent picks an action (epsilon-greedy restricted to valid actions).
            action = agent.select_action(state, valid_actions)
            
            # Environment processes the action.
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # If not done, get valid actions for the next state; otherwise empty list.
            next_valid_actions = env.get_valid_action() if not done else []
            
            # Store transition in replay memory.
            agent.push_memory((state, action, reward, next_state, done, next_valid_actions))
            
            # Optimize (update) the policy network using a minibatch from memory.
            agent.optimize_model()
            
            # Move to the next state.
            state = next_state
            total_steps += 1
            
            # Periodically update the target network with policy_net weights.
            if total_steps % agent.target_update == 0:
                agent.update_target()
        
        #  Decay epsilon AFTER the episode ends, not after every batch update
        try : 
            if agent.epsilon > agent.epsilon_end and (episode % (num_episodes // 600)==0):
                agent.epsilon *= agent.epsilon_decay  
        except ZeroDivisionError:
            pass
        # Print training progress every 'print_interval' episodes.
        if (episode + 1) % print_interval == 0:
            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, Score: {env.scorecard[0:6]}|{env.scorecard[6:]},total : {np.sum(env.scorecard)}, Epsilon: {agent.epsilon:.3f}")

    # After training, optionally save the agent.
    if save_filepath is not None:
        save_agent(agent, save_filepath)

    return agent

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
        line = f"| {steps} | **{dice_str}** | {next_state[30]} | {next_state[31]} | `{next_state[32:43].astype(int)}` | **{category[action]}** | {reward:.2f} | {cumulative_reward:.2f} | {done} |"
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




if __name__ == "__main__":

    # Just a debug check: prints True if GPU is available
    print("CUDA available?", torch.cuda.is_available())
    #print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.device("cpu")

    # Number of training episodes for this run
    num_episodes = 1000

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
    trained_agent = train_agent(
        num_episodes=num_episodes,
        print_interval=50,
        load_filepath=load_filepath,      # Might be None if not found
        save_filepath=save_filepath, lr=5e-5, gamma=0.99,
                 epsilon_start=1.0 / (trial+2), epsilon_end=0.010, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=128, target_update=100, rng=rng
    )

    # 4) Test the agent
    test_score = test_agent(trained_agent, num_test_episodes=100)
    print(f"Final average test reward: {test_score:.2f}")
    play_episode(trained_agent)

    # 5) save the model info in modelinfo.md
    save_model_info(trained_agent.policy_net.net, trained_agent.optimizer, nn.MSELoss())