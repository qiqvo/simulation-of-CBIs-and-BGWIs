from branching_processes_simulation.random_process.stable_cb import StableCB

# from branching_processes_simulation.random_process.stable_cbi import StableCBI


def main():
    alpha = 0.9
    c = 0.5
    # d = 0.5
    # d = alpha * c * (1 - 0.5)
    X = StableCB(alpha, c)
    # Z = StableCBI(alpha, c, d)

    N = 20
    t = 40
    z = 10
    # s = X.sample(1000, t, [z])
    # print(s)
    # plt.hist(s, bins=10)
    # plt.title("Histogram of Samples from StableCB")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # # plt.grid()
    # plt.show()

    # X.plot_profile(N, t, z, t_per_1=5)
    fig, _, _ = X.animate_profile(N, t, z, t_per_1=5)
    fig.savefig(f"./images/profile_{X}.png", dpi=350, bbox_inches="tight")

    # fig, _, _ = Z.animate_profile(N, t, z, t_per_1=5)
    # fig.savefig(f"./images/profile_{Z}.png", dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    main()
