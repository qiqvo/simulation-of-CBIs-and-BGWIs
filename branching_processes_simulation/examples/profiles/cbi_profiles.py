from branching_processes_simulation.random_process.plots import plot_profile, create_fig_with_axis
from branching_processes_simulation.random_process.stable_cbi import StableCBI


def main():
    # Create a StableCB process
    alpha = 0.6
    c = 1
    d = 0.3
    Z = StableCBI(alpha, c, d)
    print(Z.delta)

    # Define time and z values
    time = 2.0
    z = 3.0
    N = 20

    # Sample from the process
    times, profiles = Z.sample_profile(N, time, z, t_per_1=50)

    # Plot the results
    fig = create_fig_with_axis(times, profiles)
    for profile in profiles:
        fig = plot_profile(fig, times, profile)
    fig.show()


if __name__ == "__main__":
    main()