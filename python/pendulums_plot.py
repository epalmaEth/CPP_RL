import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import argparse


def translate(x, y, dx, dy=0):
    """Translate (x, y) coordinates by (dx, dy)."""
    x_new = x + dx
    y_new = y + dy
    return x_new, y_new


def rotate(x, y, theta):
    """Rotate (x, y) coordinates by angle theta."""
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new


def setup_plot():
    """Create the figure and axes for the pendulum animation and additional graphs."""
    fig = plt.figure(figsize=(10, 5))
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan=10)
    ax_action = plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan=5)
    ax_reward = plt.subplot2grid((10, 10), (5, 5), colspan=5, rowspan=5)
    plt.subplots_adjust(hspace=0.5, wspace=1.0)
    return fig, ax_main, ax_action, ax_reward


def setup_axes(ax_main, bound):
    """Set up the main axis limits, aspect, and labels."""
    ax_main.set_xlim(-bound, bound)
    ax_main.set_ylim(-bound, bound)
    ax_main.set_aspect("equal", "box")
    ax_main.set_xlabel("X Position")
    ax_main.set_ylabel("Y Position")
    ax_main.set_title("Pendulum Motion")


def setup_action_reward_axes(ax_action, ax_reward, actions, rewards):
    """Set up the auxiliary plots for action and reward over time."""
    (action_line,) = ax_action.plot([], [], label="Action", color="green")
    ax_action.set_xlim(0, len(actions))
    ax_action.set_ylim(np.min(actions) - 0.1, np.max(actions) + 0.1)
    ax_action.set_xticks([])
    ax_action.legend()

    (reward_line,) = ax_reward.plot([], [], label="Reward", color="blue")
    ax_reward.set_xlim(0, len(rewards))
    ax_reward.set_ylim(np.min(rewards) - 0.1, np.max(rewards) + 0.1)
    ax_reward.set_xlabel("Time Step")
    ax_reward.legend()
    return action_line, reward_line


def create_cart(ax, cart_length, cart_width, color):
    """Create and add a cart polygon to the axis."""
    cart_coords = [
        (-cart_length / 2, -cart_width / 2),
        (-cart_length / 2, cart_width / 2),
        (cart_length / 2, cart_width / 2),
        (cart_length / 2, -cart_width / 2),
    ]
    cart = Polygon(
        cart_coords, closed=True, fill=True, edgecolor=color, facecolor=color
    )
    cart.set_animated(True)
    ax.add_patch(cart)
    return cart, cart_coords


def create_rod(ax, rod_length, rod_width, color):
    """Create a rod polygon and its pivot and endpoint circles.
    The rod is defined in local coordinates (origin at pivot along x-axis).
    """
    rod_coords = [
        (0, -rod_width / 2),
        (0, rod_width / 2),
        (rod_length, rod_width / 2),
        (rod_length, -rod_width / 2),
    ]
    rod_poly = Polygon(
        rod_coords, closed=True, fill=True, edgecolor=color, facecolor=color
    )
    rod_poly.set_animated(True)
    ax.add_patch(rod_poly)

    pivot_circle = Circle((0.0, 0.0), rod_width / 2, edgecolor=color, facecolor=color)
    pivot_circle.set_animated(True)
    ax.add_patch(pivot_circle)

    end_circle = Circle(
        (rod_length, 0), rod_width / 2, edgecolor=color, facecolor=color
    )
    end_circle.set_animated(True)
    ax.add_patch(end_circle)

    return rod_poly, pivot_circle, end_circle, rod_coords


def setup_pendulum_elements(
    ax, cart_length, cart_width, rod_length, rod_width, num_rods, use_cart
):
    """Create the cart (if applicable) and a chain of rod elements.
    Returns the cart (or None), the list of rods, pivots, endpoints, their local coordinates, and the base marker.
    """
    cart_color = (129 / 255, 132 / 255, 203 / 255)
    rod_color = (204 / 255, 77 / 255, 77 / 255)

    if use_cart:
        cart, cart_coords = create_cart(ax, cart_length, cart_width, cart_color)
    else:
        cart, cart_coords = None, None

    # Update current_pivot for next rod (will be updated in animation)
    # Here we initialize current_pivot at base; updates occur in update_all_rods.
    rods = []
    pivots = []  # circles at the joints
    ends = []  # circles at rod endpoints
    rods_coords = []
    for _ in range(num_rods):
        rod_poly, pivot_circle, end_circle, rod_coords = create_rod(
            ax, rod_length, rod_width, rod_color
        )
        rods.append(rod_poly)
        pivots.append(pivot_circle)
        ends.append(end_circle)
        rods_coords.append(rod_coords)

    base_marker = Circle(
        (0.0, 0.0), rod_width / 3, edgecolor="black", facecolor="black"
    )
    base_marker.set_animated(True)
    ax.add_patch(base_marker)

    return cart, cart_coords, rods, pivots, ends, rods_coords, base_marker


def update_cart(cart, x_val, cart_coords):
    """Update the cart's position by shifting its coordinates."""
    new_points = [translate(x, y, x_val) for (x, y) in cart_coords]
    cart.set_xy(new_points)
    return cart


def update_all_rods(cart, rods, pivots, ends, rods_coords, x_cart, angles):
    """Update positions for all rod elements given the base x position and relative angles.
    angles is a 1D array of length num_rods.
    """
    # Determine base pivot
    base_pivot = (0, 0) if cart is None else (x_cart, 0)

    current_pivot = base_pivot
    cumulative_angle = 0
    for rod_coords, rod, pivot, end, angle in zip(
        rods_coords, rods, pivots, ends, angles
    ):
        cumulative_angle += angle
        global_angle = np.pi / 2 + cumulative_angle
        # Update rod polygon vertices by rotating local coordinates and translating by current pivot.
        new_points = [
            translate(*rotate(x, y, global_angle), *current_pivot)
            for x, y in rod_coords
        ]
        rod.set_xy(new_points)
        # Update pivot circle at the current joint.
        pivot.center = current_pivot
        # Update endpoint: rotate the local endpoint and translate.
        rod_length = rod_coords[-1][0]
        end_point = translate(*rotate(rod_length, 0, global_angle), *current_pivot)
        end.center = end_point
        # Next rod’s pivot is the current rod’s end
        current_pivot = end_point


def update_frame(
    frame,
    cart,
    cart_coords,
    rods,
    pivots,
    ends,
    rods_coords,
    base_marker,
    action_line,
    reward_line,
    x_cart,
    angles,
    actions=None,
    rewards=None,
):
    """Initialize animation by updating all elements to the first frame."""
    # Update cart if applicable
    if cart is not None:
        update_cart(cart, x_cart[frame], cart_coords)
        base_marker.center = x_cart[frame], 0
    update_all_rods(
        cart,
        rods,
        pivots,
        ends,
        rods_coords,
        x_cart[frame],
        angles[frame],
    )

    if frame == 0:
        action_line.set_data([], [])
        reward_line.set_data([], [])
    else:
        action_line.set_data(np.arange(frame + 1), actions[: frame + 1])
        reward_line.set_data(np.arange(frame + 1), rewards[: frame + 1])

    if cart is None:
        return rods + pivots + ends + [action_line, reward_line]
    return [cart] + rods + pivots + ends + [action_line, reward_line]


def save_animation(ani, filename):
    """Save the animation as a video file."""
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
    print("-------Saving Video-------")
    ani.save(filename, writer=writer)


def main():
    parser = argparse.ArgumentParser(description="Multi-Rod Pendulum Animation")
    parser.add_argument(
        "--task",
        type=str,
        help="Task name for the pendulum simulation (e.g., 'cartpole', 'pendulum')",
    )
    parser.add_argument(
        "--run_id", type=int, default=1, help="Run ID for the animation"
    )
    parser.add_argument(
        "--rod_length", type=float, default=1.0, help="Length of each rod element"
    )
    parser.add_argument(
        "--rod_width", type=float, default=0.2, help="Width of each rod element"
    )
    parser.add_argument(
        "--cart_length", type=float, default=0.2, help="Length of the cart"
    )
    parser.add_argument(
        "--cart_width", type=float, default=0.1, help="Width of the cart"
    )
    parser.add_argument(
        "--bound", type=float, default=2.2, help="Axis bounds for the plot"
    )
    parser.add_argument(
        "--num_rods", type=int, default=1, help="Number of rod elements in the pendulum"
    )
    parser.add_argument(
        "--use_cart", action="store_true", help="Enable cart-based motion"
    )
    args = parser.parse_args()

    # Construct run path and load data.
    # CSV file is assumed to have a header row.
    # Format:
    # For cart-based: theta1, ..., thetaN, x_cart, action, reward
    # For fixed-base: theta1, ..., thetaN, action, reward
    run_path = f"data/{args.task}/run_{args.run_id}"
    data = np.loadtxt(run_path + "/trajectory.csv", delimiter=",", skiprows=1)

    # Determine column offset for cart mode
    offset = 1 if args.use_cart else 0
    expected_cols = offset + args.num_rods + 2  # offset + num_rods + action + reward
    if data.shape[1] != expected_cols:
        mode = "cart-based" if args.use_cart else "fixed-base"
        raise ValueError(
            f"Expected {expected_cols} columns for {mode} mode, got {data.shape[1]}"
        )

    # Extract data based on the offset
    angles = data[:, : args.num_rods]
    x_cart = data[:, args.num_rods] if args.use_cart else np.zeros(data.shape[0])
    actions = data[:, offset + args.num_rods]
    rewards = data[:, offset + args.num_rods + 1]

    # Ensure that angles is a 2D array (time_steps x num_rods)
    angles = angles.reshape(-1, args.num_rods)

    # Set up the plot
    fig, ax_main, ax_action, ax_reward = setup_plot()
    if args.use_cart:
        ax_main.axhline(0, color="black", linewidth=1, alpha=0.7)
    setup_axes(ax_main, args.bound)
    action_line, reward_line = setup_action_reward_axes(
        ax_action, ax_reward, actions, rewards
    )

    # Set up pendulum elements
    cart, cart_coords, rods, pivots, ends, rods_coords, base_marker = (
        setup_pendulum_elements(
            ax_main,
            args.cart_length,
            args.cart_width,
            args.rod_length,
            args.rod_width,
            args.num_rods,
            args.use_cart,
        )
    )

    init_func = lambda: update_frame(
        0,
        cart,
        cart_coords,
        rods,
        pivots,
        ends,
        rods_coords,
        base_marker,
        action_line,
        reward_line,
        x_cart,
        angles,
    )
    update_func = lambda frame: update_frame(
        frame,
        cart,
        cart_coords,
        rods,
        pivots,
        ends,
        rods_coords,
        base_marker,
        action_line,
        reward_line,
        x_cart,
        angles,
        actions,
        rewards,
    )

    # Create the animation object
    ani = animation.FuncAnimation(
        fig,
        update_func,
        frames=len(angles),
        init_func=init_func,
        blit=True,
    )

    # Save the animation
    save_animation(ani, run_path + "/trajectory.mp4")


if __name__ == "__main__":
    main()
