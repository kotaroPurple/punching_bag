import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def bessel_product_sum(z, k, max_n=20):
    """
    Calculate the sum of J_n(z)J_{n-k}(z) from n=max(0,k) to infinity
    
    Parameters:
    -----------
    z : float
        Argument of Bessel functions
    k : int
        Difference in indices (n-m)
    max_n : int
        Maximum n to use in approximation
    
    Returns:
    --------
    sum_value : float
        Approximated sum
    """
    sum_value = 0
    for n in range(max(0, k), max_n + 1):
        sum_value += special.jv(n, z) * special.jv(n - k, z)
    return sum_value

def verify_bessel_sum_formulas(z_values, max_k=5, max_n=50):
    """
    Verify the sum formulas for Bessel functions
    
    Parameters:
    -----------
    z_values : array
        Array of z values to test
    max_k : int
        Maximum k to test
    max_n : int
        Maximum n to use in approximation
    """
    results = {}
    
    for z in z_values:
        results[z] = {}
        
        # Verify sum of squares equals 1
        sum_squares = sum(special.jv(n, z)**2 for n in range(-max_n, max_n+1))
        results[z]['sum_squares'] = sum_squares
        
        # Verify sums for different k values
        for k in range(1, max_k+1):
            sum_k = sum(special.jv(n, z) * special.jv(n-k, z) for n in range(-max_n, max_n+1))
            results[z][f'sum_k_{k}'] = sum_k
    
    # Print results
    print("Verification of Bessel function sum formulas:")
    print("--------------------------------------------")
    
    for z in z_values:
        print(f"\nz = {z}:")
        print(f"  Sum J_n^2(z) = {results[z]['sum_squares']:.8f} (should be 1)")
        
        for k in range(1, max_k+1):
            print(f"  Sum J_n(z)J_n-{k}(z) = {results[z][f'sum_k_{k}']:.8f} (should be 0)")

def plot_bessel_products(z, k_values, max_n=20):
    """
    Plot the products J_n(z)J_{n-k}(z) for different k values
    
    Parameters:
    -----------
    z : float
        Argument of Bessel functions
    k_values : list
        List of k values to plot
    max_n : int
        Maximum n to plot
    """
    plt.figure(figsize=(10, 6))
    
    for k in k_values:
        products = []
        n_values = list(range(max(0, k), max_n + 1))
        
        for n in n_values:
            products.append(special.jv(n, z) * special.jv(n - k, z))
        
        plt.plot(n_values, products, 'o-', label=f'k = {k}')
    
    plt.xlabel('n')
    plt.ylabel('J_n(z)J_{n-k}(z)')
    plt.title(f'Products of Bessel functions for z = {z}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bessel_sum_vs_cos(z, omega, t, k=1):
    """
    Compare the sum 2∑J_n(z)J_{n-k}(z)cos(kωt) with analytical formulas
    
    Parameters:
    -----------
    z : float
        Argument of Bessel functions
    omega : float
        Angular frequency
    t : array
        Time array
    k : int
        Difference in indices (n-m)
    """
    # Calculate the sum numerically
    sum_value = bessel_product_sum(z, k)
    numerical = 2 * sum_value * np.cos(k * omega * t)
    
    # Calculate analytical approximation
    if k == 1:
        analytical = (z/2) * np.cos(omega * t)
    elif k == 2:
        analytical = (z**2/8) * np.cos(2 * omega * t)
    else:
        analytical = np.zeros_like(t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, numerical, 'b-', label=f'Numerical: 2∑J_n(z)J_n-{k}(z)cos({k}ωt)')
    plt.plot(t, analytical, 'r--', label=f'Analytical approximation')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Comparison for z = {z}, k = {k}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Verify sum formulas
    z_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    verify_bessel_sum_formulas(z_values)
    
    # Plot products for z = 1
    plot_bessel_products(1.0, [1, 2, 3])
    
    # Compare sum with cos for different k values
    z = 0.5
    omega = 2 * np.pi
    t = np.linspace(0, 1, 1000)
    
    for k in [1, 2]:
        plot_bessel_sum_vs_cos(z, omega, t, k)

if __name__ == "__main__":
    main()