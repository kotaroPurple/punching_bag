import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def verify_bessel_sum_formula_k1(z_values, max_n=50):
    """
    Verify the formula: 2∑J_n(z)J_{n-1}(z)cos(ωt) = (z/2)cos(ωt)
    
    Parameters:
    -----------
    z_values : array
        Array of z values to test
    max_n : int
        Maximum n to use in approximation
    """
    results = {}
    
    for z in z_values:
        # Calculate the sum numerically
        sum_value = 0
        for n in range(1, max_n + 1):
            sum_value += special.jv(n, z) * special.jv(n-1, z)
        
        # Calculate the analytical value
        analytical = z / 4
        
        # Store results
        results[z] = {
            'numerical': sum_value,
            'analytical': analytical,
            'error': abs(sum_value - analytical) / analytical * 100  # percent error
        }
    
    # Print results
    print("Verification of formula: 2∑J_n(z)J_{n-1}(z)cos(ωt) = (z/2)cos(ωt)")
    print("------------------------------------------------------------")
    print(f"{'z':^10}{'Numerical':^15}{'Analytical':^15}{'Error (%)':^15}")
    print("-" * 55)
    
    for z in z_values:
        print(f"{z:^10.2f}{results[z]['numerical']:^15.8f}{results[z]['analytical']:^15.8f}{results[z]['error']:^15.8f}")
    
    return results

def verify_bessel_sum_formula_k2(z_values, max_n=50):
    """
    Verify the formula: 2∑J_n(z)J_{n-2}(z)cos(2ωt) = (z²/8)cos(2ωt)
    
    Parameters:
    -----------
    z_values : array
        Array of z values to test
    max_n : int
        Maximum n to use in approximation
    """
    results = {}
    
    for z in z_values:
        # Calculate the sum numerically
        sum_value = 0
        for n in range(2, max_n + 1):
            sum_value += special.jv(n, z) * special.jv(n-2, z)
        
        # Calculate the analytical value
        analytical = z**2 / 16
        
        # Store results
        results[z] = {
            'numerical': sum_value,
            'analytical': analytical,
            'error': abs(sum_value - analytical) / analytical * 100 if analytical != 0 else float('inf')  # percent error
        }
    
    # Print results
    print("\nVerification of formula: 2∑J_n(z)J_{n-2}(z)cos(2ωt) = (z²/8)cos(2ωt)")
    print("--------------------------------------------------------------")
    print(f"{'z':^10}{'Numerical':^15}{'Analytical':^15}{'Error (%)':^15}")
    print("-" * 55)
    
    for z in z_values:
        print(f"{z:^10.2f}{results[z]['numerical']:^15.8f}{results[z]['analytical']:^15.8f}{results[z]['error']:^15.8f}")
    
    return results

def plot_bessel_sum_vs_analytical(z_values, omega, t):
    """
    Plot the numerical sum vs analytical formula for different z values
    
    Parameters:
    -----------
    z_values : array
        Array of z values to plot
    omega : float
        Angular frequency
    t : array
        Time array
    """
    fig, axs = plt.subplots(len(z_values), 2, figsize=(12, 4*len(z_values)))
    
    for i, z in enumerate(z_values):
        # k=1 case
        sum_k1 = 0
        for n in range(1, 50):
            sum_k1 += special.jv(n, z) * special.jv(n-1, z)
        
        numerical_k1 = 2 * sum_k1 * np.cos(omega * t)
        analytical_k1 = (z/2) * np.cos(omega * t)
        
        axs[i, 0].plot(t, numerical_k1, 'b-', label='Numerical')
        axs[i, 0].plot(t, analytical_k1, 'r--', label='Analytical')
        axs[i, 0].set_title(f'k=1, z={z}')
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Amplitude')
        axs[i, 0].legend()
        axs[i, 0].grid(True)
        
        # k=2 case
        sum_k2 = 0
        for n in range(2, 50):
            sum_k2 += special.jv(n, z) * special.jv(n-2, z)
        
        numerical_k2 = 2 * sum_k2 * np.cos(2 * omega * t)
        analytical_k2 = (z**2/8) * np.cos(2 * omega * t)
        
        axs[i, 1].plot(t, numerical_k2, 'b-', label='Numerical')
        axs[i, 1].plot(t, analytical_k2, 'r--', label='Analytical')
        axs[i, 1].set_title(f'k=2, z={z}')
        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Amplitude')
        axs[i, 1].legend()
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def verify_dc_removed_intensity(z, omega, t, max_n=20):
    """
    Verify the formula for DC-removed intensity
    
    Parameters:
    -----------
    z : float
        Argument of Bessel functions (2kd0)
    omega : float
        Angular frequency
    t : array
        Time array
    max_n : int
        Maximum n to use in approximation
    """
    # Calculate the full signal
    s = np.zeros(len(t), dtype=complex)
    for n in range(-max_n, max_n+1):
        s += special.jv(n, z) * np.exp(1j * n * omega * t)
    
    # Calculate DC component
    J0 = special.jv(0, z)
    
    # Calculate DC-removed intensity directly
    intensity_direct = np.abs(s - J0)**2
    
    # Calculate using the formula
    intensity_formula = 1 + J0**2 - 2*J0*np.cos(z*np.sin(omega*t))
    
    # Calculate using the expanded formula
    intensity_expanded = 1 - J0**2
    for n in range(1, max_n+1):
        intensity_expanded -= 4*J0*special.jv(2*n, z)*np.cos(2*n*omega*t)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, intensity_direct, 'b-', label='Direct calculation')
    plt.plot(t, intensity_formula, 'r--', label='Formula: 1 + J0² - 2J0cos(z·sin(ωt))')
    plt.plot(t, intensity_expanded, 'g-.', label='Expanded formula')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.title(f'DC-removed intensity for z = {z}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Test values
    z_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    omega = 2 * np.pi
    t = np.linspace(0, 1, 1000)
    
    # Verify formulas
    verify_bessel_sum_formula_k1(z_values)
    verify_bessel_sum_formula_k2(z_values)
    
    # Plot comparisons
    plot_bessel_sum_vs_analytical(z_values[:3], omega, t)
    
    # Verify DC-removed intensity formula
    verify_dc_removed_intensity(1.0, omega, t)

if __name__ == "__main__":
    main()