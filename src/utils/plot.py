import pandas as pd
import matplotlib.pyplot as plt


def plot_mae_over_time(df, output_dir='plots'):
    """
    Plots the Mean Absolute Error (MAE) over time and saves the figure to the specified output directory.

    Args:
        mae_df (pd.DataFrame): DataFrame containing 'time' and 'mae' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['mae'], marker='o', linestyle='-', color='b')
    plt.title('Mean Absolute Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.savefig(f'{output_dir}/mae_over_time.png')
    plt.close()

def plot_test_metrics(test_df, output_dir='plots'):
    """
    Plots the test metrics and saves the figure to the specified output directory.

    Args:
        test_df (pd.DataFrame): DataFrame containing test inputs, targets and predictions.
    """

    
    
    # Plot MAE over time
    plot_mae_over_time(test_df, output_dir)




