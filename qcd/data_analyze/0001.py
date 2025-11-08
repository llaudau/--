import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
FILE_PATH = "/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/actiontest5.txt"  # <-- Set your file name here
NUM_LINES_TO_READ = 1000
PLOT_FILENAME = "/home/khw/Documents/Git_repository/qcd/data_analyze/05.png"
NUM_BINS = 40  # Number of bars in the histogram

def plot_action_distribution_numpy_only(filepath, num_lines, plot_filename, num_bins):
    """Reads double numbers from a file using standard I/O and plots their distribution."""

    data = []

    # 1. Read the data using standard Python file I/O
    try:
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if i >= num_lines:
                    break  # Stop after reading the required number of lines
                
                try:
                    # Convert the stripped line (removing newline characters) to a float
                    data.append(float(line.strip()))
                except ValueError:
                    print(f"Warning: Skipping non-numeric data on line {i+1}.")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    if not data:
        print(f"Successfully read 0 lines. Check if '{filepath}' is empty or formatted correctly.")
        return

    print(f"Successfully read {len(data)} action values.")

    # Convert the list to a NumPy array for efficient calculation
    data_array = np.array(data)
    
    # 2. Calculate Key Statistics using NumPy
    mean_action = np.mean(data_array)
    std_action = np.std(data_array)

    # 3. Create the Histogram Figure
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(
        data_array, 
        bins=num_bins, 
        color='lightcoral', 
        edgecolor='black', 
        alpha=0.7
    )
    
    # 4. Add Labels, Title, and Stats
    plt.title('Distribution of Lattice Action Values ($\propto e^{-S}$)', fontsize=14)
    plt.xlabel(f'Action Value ($S$) [Mean: {mean_action:.2f}, Std Dev: {std_action:.2f}]', fontsize=12)
    plt.ylabel('Frequency (Configuration Count)', fontsize=12)
    
    # Add a text box with key statistics
    stats_text = (
        f'N = {len(data)}\n'
        f'Mean = {mean_action:.3f}\n'
        f'Std Dev = {std_action:.3f}'
    )
    plt.gca().text(
        0.05, 0.95, stats_text, 
        transform=plt.gca().transAxes,
        fontsize=10, 
        verticalalignment='top', 
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6)
    )
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 5. Save the figure
    plt.savefig(plot_filename)
    print(f"Histogram saved as {plot_filename}")

# --- Execution ---
if __name__ == "__main__":
    plot_action_distribution_numpy_only(FILE_PATH, NUM_LINES_TO_READ, PLOT_FILENAME, NUM_BINS)