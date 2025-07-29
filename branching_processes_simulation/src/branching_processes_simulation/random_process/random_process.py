from abc import abstractmethod
from typing import List
import numpy as np

from branching_processes_simulation.i_random import IRandom


class RandomProcess(IRandom):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @abstractmethod
    def characteristic_function(
        self, t: np.complex64, time: float, z: np.float64
    ) -> np.complex64:
        return None

    @abstractmethod
    def laplace_transform(
        self, t: np.float64, time: float, z: np.float64
    ) -> np.float64:
        return None

    @abstractmethod
    def mean(self, time: float, z: np.float64) -> np.float64:
        return None

    @abstractmethod
    def variance(self, time: float, z: np.float64) -> np.float64:
        return None

    ## Returns a sample in a shape (len(z), N)
    @abstractmethod
    def sample(
        self, N: int, time: np.float64, z: List[np.float64], **kwargs
    ) -> np.ndarray[np.ndarray[np.float64]]:
        return None

    @abstractmethod
    def _get_profile_times(self, time, **kwargs):
        return None

    ## Returns times and a sample in a shape (N, len(times)), len(times) ~ t_per_1 * time
    def sample_profile(
        self, N: int, time: float, z: float, **kwargs
    ) -> np.ndarray[int]:
        times = self._get_profile_times(time, **kwargs)
        m = len(times)

        profile = np.zeros((N, m), np.float64)
        profile[:, 0] = [z] * N
        for i in range(1, m):
            dt = times[i]
            profile[:, i] = self.sample(1, dt, profile[:, i - 1], **kwargs)[:, 0]

        return times, profile

    def plot_profile(self, N: int, time: float, z: float, **kwargs) -> None:
        import matplotlib.pyplot as plt

        times, profile = self.sample_profile(N, time, z, **kwargs)
        for i in range(N):
            plt.plot(times, profile[i], label=f"Sample {i + 1}")

        plt.title(f"Profile of {self}")
        plt.xlabel("Time")
        plt.ylabel("Value")

    def animate_profile(
        self, N: int, time: float, z: float, folder="./images", **kwargs
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        times, profile = self.sample_profile(N, time, z, **kwargs)

        fig, ax = plt.subplots()
        lines = [ax.plot([], [], lw=2)[0] for _ in range(N)]
        ax.grid()
        ax.set_title(f"Profile of {self}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(times[:frame], profile[i, :frame])

            ax.relim()
            ax.autoscale_view()

            return lines

        ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False)
        fps = kwargs.get("fps", kwargs.get("t_per_1", 5))
        ani.save(
            folder + f"/profile_animation_{self}.gif",
            writer=PillowWriter(fps=fps),
            dpi=350,
        )
        return fig, ax, ani
