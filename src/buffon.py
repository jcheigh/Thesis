import numpy as np
import matplotlib.pyplot as plt
def simulate_2n_cross_intersections(trials, n):
    """
    Simulates the intersections of a 2n-cross with parallel lines.
    
    Parameters:
    trials (int): Number of trials to simulate
    n (int): Number of sticks in the cross (2n sticks)
    
    Returns:
    list: A list containing counts of 0, 1, ..., 2n intersections
    """
    results = [0] * (2 * n + 1)  # [0 intersections, 1 intersection, ..., 2n intersections]

    for _ in range(trials):
        # Generate random position for the center of the cross (0 to 1)
        y_center = np.random.uniform(0, 1)
        
        # Generate random angle for the first needle (0 to pi)
        angle = np.random.uniform(0, np.pi)
        
        intersections = 0
        for i in range(2 * n):
            # Calculate the angle for each stick in the cross
            current_angle = angle + (i * np.pi / n)
            
            # Calculate the vertical distance from the center of the stick to the nearest line
            vertical_distance = np.abs(np.sin(current_angle)) / 2
            
            # Check if the stick intersects a line
            if y_center <= vertical_distance or y_center >= 1 - vertical_distance:
                intersections += 1
        
        results[intersections] += 1
    
    return results

def main(n):
    trials = 1000000  # Number of trials for the simulation
    results = simulate_2n_cross_intersections(trials, n)
    
    print(f"Results after {trials} trials:")
    for i in range(2 * n + 1):
        print(f"{i} intersections: {results[i]} ({results[i] / trials * 100:.2f}%)")
    
    total_intersections = sum(i * results[i] for i in range(2 * n + 1)) / trials
    print(f"Normalized total count of sticks intersected: {total_intersections:.4f}")
    
    # Plot the simulation results
    labels = [f'{i} Intersections' for i in range(2 * n + 1)]
    percentages = [f'{results[i] / trials * 100:.2f}%' for i in range(2 * n + 1)]
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, results, color='skyblue', alpha=0.7)
    plt.title("Distribution of Intersections in Buffon's 2n-Cross Problem")
    plt.xlabel("Number of Intersections")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Add percentage labels on top of the bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, percentage, ha='center', va='bottom')
    
    plt.show()

if __name__ == "__main__":
    import numpy as np
    from scipy.integrate import dblquad

    # Define the parameters
    l = 1.0  # length of the needle
    t = 1.0  # distance between lines
    alpha1 = np.pi / 4  # angle between the first and second sticks
    alpha2 = np.pi / 4  # angle between the second and third sticks

    # Define the integrand
    def integrand(x, theta, l, alpha1, alpha2):
        return 4 / (np.pi * t)

    # Integration limits for 3 intersections
    def upper_limit_3(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta), l / 2 * np.sin(theta + alpha1), l / 2 * np.sin(theta + alpha1 + alpha2))

    # Calculate the probability of 3 intersections
    P_3_intersections, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: 0,
        lambda theta: upper_limit_3(theta, l, alpha1, alpha2),
        args=(l, alpha1, alpha2)
    )

    # Integration limits for 2 intersections (three separate cases)
    def upper_limit_2a(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta), l / 2 * np.sin(theta + alpha1 + alpha2))

    def lower_limit_2a(theta, l, alpha1, alpha2):
        return l / 2 * np.sin(theta + alpha1)

    P_2a, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: lower_limit_2a(theta, l, alpha1, alpha2),
        lambda theta: upper_limit_2a(theta, l, alpha1, alpha2),
        args=(l, alpha1, alpha2)
    )

    def upper_limit_2b(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta + alpha1), l / 2 * np.sin(theta + alpha1 + alpha2))

    def lower_limit_2b(theta, l, alpha1, alpha2):
        return l / 2 * np.sin(theta)

    P_2b, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: lower_limit_2b(theta, l, alpha1, alpha2),
        lambda theta: upper_limit_2b(theta, l, alpha1, alpha2),
        args=(l, alpha1, alpha2)
    )

    def upper_limit_2c(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta), l / 2 * np.sin(theta + alpha1))

    def lower_limit_2c(theta, l, alpha1, alpha2):
        return l / 2 * np.sin(theta + alpha1 + alpha2)

    P_2c, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: lower_limit_2c(theta, l, alpha1, alpha2),
        lambda theta: upper_limit_2c(theta, l, alpha1),
        args=(l, alpha1, alpha2)
    )

    P_2_intersections = P_2a + P_2b + P_2c

    # Integration limits for 1 intersection (three separate cases)
    def upper_limit_1a(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta), l / 2 * np.sin(theta + alpha1 + alpha2))

    def lower_limit_1a(theta, l, alpha1, alpha2):
        return max(l / 2 * np.sin(theta + alpha1), l / 2 * np.sin(theta + alpha1 + alpha2))

    P_1a, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: lower_limit_1a(theta, l, alpha1, alpha2),
        lambda theta: upper_limit_1a(theta, l, alpha1, alpha2),
        args=(l, alpha1, alpha2)
    )

    def upper_limit_1b(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta + alpha1), l / 2 * np.sin(theta + alpha1 + alpha2))

    def lower_limit_1b(theta, l, alpha1, alpha2):
        return max(l / 2 * np.sin(theta), l / 2 * np.sin(theta + alpha1 + alpha2))

    P_1b, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: lower_limit_1b(theta, l, alpha1, alpha2),
        lambda theta: upper_limit_1b(theta, l, alpha1 + alpha2),
        args=(l, alpha1, alpha2)
    )

    def upper_limit_1c(theta, l, alpha1, alpha2):
        return min(l / 2 * np.sin(theta), l / 2 * np.sin(theta + alpha1))

    def lower_limit_1c(theta, l, alpha1, alpha2):
        return max(l / 2 * np.sin(theta + alpha1 + alpha2), l / 2 * np.sin(theta + alpha1))

    P_1c, _ = dblquad(
        integrand, 0, np.pi / 2,
        lambda theta: lower_limit_1c(theta, l, alpha1, alpha2),
        lambda theta: upper_limit_1c(theta, l, alpha1),
        args=(l, alpha1, alpha2)
    )

    P_1_intersection = P_1a + P_1b + P_1c

    # Calculate the probability of 0 intersections
    P_0_intersections = 1 - P_1_intersection - P_2_intersections - P_3_intersections

    # Print the results
    print(P_0_intersections) 
    print(P_1_intersection)
    print(P_2_intersections)
    print(P_3_intersections)
