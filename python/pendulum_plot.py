import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle

def rotate(x, y, theta):
    """Rotate (x, y) coordinates by angle theta."""
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new

def setup_plot():
    """Create the figure and axis for the pendulum animation."""
    fig = plt.figure(figsize=(10, 5))
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan=10)
    ax_action = plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan=5)
    ax_reward = plt.subplot2grid((10, 10), (5, 5), colspan=5, rowspan=5)
    plt.subplots_adjust(hspace=1.0, wspace=1.0)
    return fig, ax_main, ax_action, ax_reward

def setup_pendulum_elements(ax_main, theta, rod_length, rod_width):
    """Set up the pendulum elements on the plot."""
    coords = [(0, -rod_width / 2), (0, rod_width / 2), 
              (rod_length, rod_width / 2), (rod_length, -rod_width / 2)]
    red = (204 / 255, 77 / 255, 77 / 255)
    black = (0, 0, 0)

    # Rotate coordinates for initial position
    transformed_coords = [rotate(x, y, theta[0] + np.pi / 2) for x, y in coords]
    
    rod = Polygon(transformed_coords, closed=True, fill=True, edgecolor=red, facecolor=red)
    rod_pivot = Circle((0, 0), rod_width / 2, edgecolor=red, facecolor=red)
    rod_end = Circle(rotate(rod_length, 0, theta[0] + np.pi / 2), rod_width / 2, edgecolor=red, facecolor=red)
    pivot = Circle((0, 0), 0.05, edgecolor=black, facecolor=black)
    
    ax_main.add_patch(rod)
    ax_main.add_patch(rod_pivot)
    ax_main.add_patch(rod_end)
    ax_main.add_patch(pivot)

    return rod, rod_pivot, rod_end, pivot

def setup_pendulum_ax(ax_main, bound):
    """Set up the pendulum axis with limits."""
    ax_main.set_xlim(-bound, bound)
    ax_main.set_ylim(-bound, bound)
    ax_main.set_aspect('equal', 'box')
    ax_main.set_xlabel('X Position')
    ax_main.set_ylabel('Y Position')
    ax_main.set_title('Pendulum Motion')

def setup_action_reward_axes(ax_action, ax_reward, actions, rewards):
    """Set up the action and reward axes for time-varying graphs."""
    action_line, = ax_action.plot([], [], label="Action", color='green')
    ax_action.set_xlim(0, len(actions))
    ax_action.set_ylim(np.min(actions) - 0.1, np.max(actions) + 0.1)
    ax_action.set_title('Action')
    ax_action.legend()

    reward_line, = ax_reward.plot([], [], label="Reward", color='blue')
    ax_reward.set_xlim(0, len(rewards))
    ax_reward.set_ylim(np.min(rewards) - 0.1, np.max(rewards) + 0.1)
    ax_reward.set_title('Reward')
    ax_reward.set_xlabel('Time Step')
    ax_reward.legend()

    return action_line, reward_line

def init_plot(rod, rod_end, action_line, reward_line, theta, coords):
    """Initialize the plot elements for animation."""
    transformed_coords = [rotate(x, y, theta[0] + np.pi / 2) for x, y in coords]
    rod.set_xy(transformed_coords)
    rod_end.center = rotate(1.0, 0, theta[0] + np.pi / 2)
    action_line.set_data([], [])
    reward_line.set_data([], [])
    return rod, rod_end, action_line, reward_line

def update_plot(frame, rod, rod_end, reward_line, action_line, theta, coords, actions, rewards):
    """Update the plot for each frame of the animation."""
    angle = theta[frame] + np.pi / 2
    transformed_coords = [rotate(x, y, angle) for x, y in coords]
    rod.set_xy(transformed_coords)
    rod_end.center = rotate(1.0, 0, angle)

    reward_line.set_data(np.arange(frame + 1), rewards[:frame + 1])
    action_line.set_data(np.arange(frame + 1), actions[:frame + 1])

    return rod, rod_end, reward_line, action_line

def save_animation(ani, filename):
    """Save the animation as a video file."""
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    print("-------Saving Video-------")
    ani.save(filename, writer=writer)

def main():
    """Main function to run the pendulum animation."""
    # Load the data
    data = np.loadtxt('data/render/pendulum/data.csv', delimiter=',', skiprows=1)
    theta, theta_dot, actions, rewards = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Set up the plot
    fig, ax_main, ax_action, ax_reward = setup_plot()

    # Set up pendulum elements
    rod, rod_pivot, rod_end, pivot = setup_pendulum_elements(ax_main, theta, 1.0, 0.2)

    # Set the axis limits
    setup_pendulum_ax(ax_main, 2.2)

    # Set up the action and reward axes
    action_line, reward_line = setup_action_reward_axes(ax_action, ax_reward, actions, rewards)

    # Initialize the plot
    init_func = lambda: init_plot(rod, rod_end, action_line, reward_line, theta, [(0, -0.1), (0, 0.1), (1, 0.1), (1, -0.1)])

    # Create the animation object
    ani = animation.FuncAnimation(fig, update_plot, frames=len(theta), fargs=(rod, rod_end, reward_line, action_line, theta, [(0, -0.1), (0, 0.1), (1, 0.1), (1, -0.1)], actions, rewards), init_func=init_func, blit=True)

    # Save the animation
    save_animation(ani, 'videos/pendulum/pendulum_motion.mp4')

if __name__ == "__main__":
    main()

