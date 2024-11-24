import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for visualization
from matplotlib.animation import FuncAnimation  # For creating animations
import os  # Import os for file system operations
from matplotlib.patches import Circle  # For drawing nodes in gradient visualization
import matplotlib
from functools import partial  # For simplifying function calls with pre-filled arguments

# Configure matplotlib to use a non-interactive backend for saving animations
matplotlib.use("Agg")

# Define the directory where results (e.g., animations) will be saved
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define a simple Multi-Layer Perceptron (MLP) class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(42)  # Set random seed for reproducibility
        self.lr = lr  # Learning rate for gradient descent
        self.activation_fn = activation  # Activation function for hidden layer

        # Initialize weights and biases for input-hidden and hidden-output layers
        self.weights_with_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        # For visualization purposes
        self.hidden_activations = None  # To store hidden layer activations
        self.gradients = None  # To store computed gradients

    def forward(self, inputs):
        # Compute hidden layer pre-activations
        z_hidden = np.dot(inputs, self.weights_with_input_hidden) + self.bias_hidden
        # Apply activation function to hidden layer
        if self.activation_fn == 'tanh':
            self.hidden_activations = np.tanh(z_hidden)
        elif self.activation_fn == 'relu':
            self.hidden_activations = np.maximum(0, z_hidden)
        elif self.activation_fn == 'sigmoid':
            self.hidden_activations = 1 / (1 + np.exp(-z_hidden))

        # Compute output layer pre-activations
        z_output = np.dot(self.hidden_activations, self.weights_hidden_output) + self.bias_output
        output = np.tanh(z_output)  # Apply tanh activation at the output
        return output

    def backward(self, inputs, labels):
        # Compute output error as the difference between prediction and true labels
        error_output = self.forward(inputs) - labels
        # Derivative of tanh activation at output layer
        d_output = error_output * (1 - self.forward(inputs) ** 2)

        # Compute gradients for hidden-output weights and biases
        weights_hidden = np.dot(self.hidden_activations.T, d_output)
        grad_output_with_bias = np.sum(d_output, axis=0, keepdims=True)

        # Backpropagate error to hidden layer
        error_hidden = np.dot(d_output, self.weights_hidden_output.T)
        if self.activation_fn == 'tanh':
            d_hidden = error_hidden * (1 - self.hidden_activations ** 2)
        elif self.activation_fn == 'relu':
            d_hidden = error_hidden * (self.hidden_activations > 0)
        elif self.activation_fn == 'sigmoid':
            d_hidden = error_hidden * (self.hidden_activations * (1 - self.hidden_activations))

        # Compute gradients for input-hidden weights and biases
        grad_weights_hidden = np.dot(inputs.T, d_hidden)
        grad_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights_with_input_hidden -= self.lr * grad_weights_hidden
        self.bias_hidden -= self.lr * grad_bias_hidden
        self.weights_hidden_output -= self.lr * weights_hidden
        self.bias_output -= self.lr * grad_output_with_bias

        # Store gradients for visualization
        self.gradients = {
            'input_hidden': grad_weights_hidden,
            'hidden_output': weights_hidden
        }

# Generate circularly distributed data
def generate_data(n_samples=100):  # Function to generate random 2D points with labels
    np.random.seed(0)  # Seed for reproducibility
    radius = np.sqrt(np.random.rand(n_samples))  # Generate radii with uniform distribution
    angles = 2 * np.pi * np.random.rand(n_samples)  # Generate angles from 0 to 2π
    X = np.c_[radius * np.cos(angles), radius * np.sin(angles)]  # Convert to Cartesian coordinates
    labels = (radius > 0.5).astype(int) * 2 - 1  # Label points as +1 or -1 based on radius
    labels = labels.reshape(-1, 1)  # Reshape labels into a column vector
    return X, labels


# Update function for animation
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Clear previous plots for the next frame
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform multiple training steps for each animation frame
    for _ in range(10):  # Update the MLP weights multiple times per frame
        mlp.forward(X)
        mlp.backward(X, y)

    # --- Hidden Layer Visualization ---
    hidden_features = mlp.hidden_activations  # Get the activations of the hidden layer
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7  # Color points by their labels
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")  # Title for the hidden space plot
    ax_hidden.set_xlabel("Hidden Dimension 1")
    ax_hidden.set_ylabel("Hidden Dimension 2")
    ax_hidden.set_zlabel("Hidden Dimension 3")

    # Plot decision hyperplane in the hidden space
    x_vals = np.linspace(-1.5, 1.5, 50)  # Range for x-axis
    y_vals = np.linspace(-1.5, 1.5, 50)  # Range for y-axis
    xx, yy = np.meshgrid(x_vals, y_vals)  # Create a grid
    z_vals = -(mlp.weights_hidden_output[0, 0] * xx +
               mlp.weights_hidden_output[1, 0] * yy +
               mlp.bias_output[0, 0]) / (mlp.weights_hidden_output[2, 0] + 1e-5)
    ax_hidden.plot_surface(xx, yy, z_vals, alpha=0.3, color='tan')  # Add the hyperplane

    # --- Input Space Decision Boundary ---
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Define range for x-axis
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Define range for y-axis
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))  # Create a grid
    grid = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid to pass into the model
    preds = mlp.forward(grid).reshape(xx.shape)  # Predict the output for the grid

    # Plot decision boundary and fill regions with different colors
    ax_input.contour(xx, yy, preds, levels=[0], colors='black', linewidths=1.5)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], colors=['red', 'blue'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', s=20)  # Plot the input data
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    ax_input.set_xlabel("Feature 1")
    ax_input.set_ylabel("Feature 2")

    # --- Gradient Visualization ---
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.set_xlim(0, 1)  # Set x-axis limits for visualization
    ax_gradient.set_ylim(0, 1)  # Set y-axis limits for visualization
    ax_gradient.axis('off')  # Hide axis lines for a cleaner look

    # Define node positions for gradient visualization
    nodes = {
        'x1': (0.2, 0.8), 'x2': (0.2, 0.6),  # Input layer nodes
        'h1': (0.5, 0.9), 'h2': (0.5, 0.7), 'h3': (0.5, 0.5),  # Hidden layer nodes
        'y': (0.8, 0.7)  # Output layer node
    }

    # Draw nodes as circles
    for name, (x, y) in nodes.items():
        ax_gradient.add_patch(Circle((x, y), 0.03, color='blue'))  # Draw a circle for each node
        ax_gradient.text(x, y, name, color='white', ha='center', va='center')  # Label each node

    # Draw edges between nodes and vary thickness by gradient magnitude
    edges = [
        ('x1', 'h1', mlp.gradients['input_hidden'][0, 0]),
        ('x1', 'h2', mlp.gradients['input_hidden'][0, 1]),
        ('x1', 'h3', mlp.gradients['input_hidden'][0, 2]),
        ('x2', 'h1', mlp.gradients['input_hidden'][1, 0]),
        ('x2', 'h2', mlp.gradients['input_hidden'][1, 1]),
        ('x2', 'h3', mlp.gradients['input_hidden'][1, 2]),
        ('h1', 'y', mlp.gradients['hidden_output'][0, 0]),
        ('h2', 'y', mlp.gradients['hidden_output'][1, 0]),
        ('h3', 'y', mlp.gradients['hidden_output'][2, 0]),
    ]

    for start, end, grad in edges:  # Loop through all edges
        x1, y1 = nodes[start]  # Start node coordinates
        x2, y2 = nodes[end]  # End node coordinates
        linewidth = min(3, max(0.5, abs(grad) * 5))  # Scale line width by gradient magnitude
        ax_gradient.plot([x1, x2], [y1, y2], 'm-', linewidth=linewidth)  # Draw the edge

# Visualization function
def visualize(activation, lr, step_num):
    # Generate training data
    X, y = generate_data()  # X is input data, y is labels
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)  # Initialize MLP model

    # Set up the figure for visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')  # Subplot for hidden layer visualization
    ax_input = fig.add_subplot(132)  # Subplot for input space visualization
    ax_gradient = fig.add_subplot(133)  # Subplot for gradient visualization

    # Create animation of the training process
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF in the results directory
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()  # Close the figure after saving to avoid display

# Main execution block
if __name__ == "__main__":
    activation = "tanh"  # Choose the activation function for the hidden layer
    lr = 0.1  # Set the learning rate
    step_num = 1000  # Total number of steps for training
    visualize(activation, lr, step_num)  # Start the visualization process
