import numpy as np
import matplotlib.pyplot as plt
from cmaes import CMA, CMAwM

def f(x, y):
    numerator = np.cos(np.sin(np.abs(x**2 - y**2)))**2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + numerator / denominator

def ellipsoid_onemax(x, n_zdim):
    n = len(x)
    n_rdim = n - n_zdim
    r = 10
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    ellipsoid = sum([(1000 ** (i / (n_rdim - 1)) * x[i]) ** 2 for i in range(n_rdim)])
    onemax = n_zdim - (0.0 < x[(n - n_zdim) :]).sum()
    return ellipsoid + r * onemax

def plot():
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Grafica la función en 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title('3D Visualization of the function f(x, y)')

    plt.show()


# real solutions:
# f(0, 1.25313) = 0.292579)
# f(0, -1.25313) = 0.292579)
# f(1.25313, 0) = 0.292579)
# f(-1.25313, 0) = 0.292579)
def main1():
    optimizer = CMA(mean=np.zeros(2), sigma=1.3, population_size=20, lr_adapt=True)

    for generation in range(100):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = f(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)

def main2():
    binary_dim, continuous_dim = 10, 10
    dim = binary_dim + continuous_dim
    bounds = np.concatenate(
        [
            np.tile([-np.inf, np.inf], (continuous_dim, 1)),
            np.tile([0, 1], (binary_dim, 1)),
        ]
    )
    steps = np.concatenate([np.zeros(continuous_dim), np.ones(binary_dim)])
    optimizer = CMAwM(mean=np.zeros(dim), sigma=2.0, bounds=bounds, steps=steps)
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = f(x_for_eval, binary_dim)
            evals += 1
            solutions.append((x_for_tell, value))
            if evals % 300 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main2()