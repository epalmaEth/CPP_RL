import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read the CSV file into a numpy array
data = np.loadtxt('data/pendulum/data.csv', delimiter=',', skiprows=1)

# Extract theta values from the data
theta = data[:, 0]  # First column is theta
theta_dot = data[:, 1]  # Second column is theta_dot
actions = data[:, 2]  # Third column is action
rewards = data[:, 3]  # Fourth column is reward

# Create a figure with a custom grid layout (main occupies 2 rows, action and reward stacked in the second column)
fig = plt.figure(figsize=(12, 6))
ax_main = plt.subplot2grid((10,10), (0,0), colspan=5, rowspan=10)
ax_action = plt.subplot2grid((10,10), (0,5), colspan=5, rowspan=5)
ax_reward = plt.subplot2grid((10,10), (5,5), colspan=5, rowspan=5)
# Adjust layout to add padding between subplots
plt.subplots_adjust(hspace=1., wspace=1.)  # Add vertical and horizontal padding between subplots

# Rod length for Cartesian coordinates
rod_length = 1.0

# Initialize the line object for the pendulum path
line, = ax_main.plot([], [], 'o-', lw=2, label='Pendulum Path')

# Set plot limits for the pendulum animation
ax_main.set_xlim(-1.2, 1.2)
ax_main.set_ylim(-1.2, 1.2)
ax_main.set_aspect('equal', 'box')
ax_main.set_xlabel('X Position')
ax_main.set_ylabel('Y Position')
ax_main.set_title('Pendulum Motion')
ax_main.legend()

# Set up the action and reward axes for time-varying graphs
ax_action.set_xlim(0, len(actions))
ax_action.set_ylim(np.min(actions) - 0.1, np.max(actions) + 0.1)
ax_action.set_title('Action')

ax_reward.set_xlim(0, len(rewards))
ax_reward.set_ylim(np.min(rewards) - 0.1, np.max(rewards) + 0.1)
ax_reward.set_title('Reward')
ax_reward.set_xlabel('Time Step')

# Initialize the lines for reward and action
reward_line, = ax_reward.plot([], [], label="Reward", color='blue')
action_line, = ax_action.plot([], [], label="Action", color='green')

# Function to initialize the plot elements
def init():
    line.set_data([], [])
    reward_line.set_data([], [])
    action_line.set_data([], [])
    return line, reward_line, action_line

# Function to update the plot for each frame of the animation
def update(frame):
    # Calculate the pendulum's position in Cartesian coordinates
    x = rod_length * np.sin(theta[frame])
    y = rod_length * np.cos(theta[frame])

    # Update the data for the pendulum (the line)
    line.set_data([0, x], [0, y])

    # Update the reward and action graphs
    reward_line.set_data(np.arange(frame + 1), rewards[:frame + 1])
    action_line.set_data(np.arange(frame + 1), actions[:frame + 1])

    return line, reward_line, action_line

# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=len(theta), init_func=init, blit=True)

# Set up the video writer
writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as a video file
print("-------Saving Video-------")
ani.save('videos/pendulum/pendulum_motion.mp4', writer=writer)

