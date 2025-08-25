"""Test backward reconstruction functionality."""

import numpy as np
import matplotlib.pyplot as plt
from svd_online_dmd import SvdOnlineDmd


def test_backward_reconstruction():
    """Test backward reconstruction capability."""
    # Create test signal
    dt = 0.01
    t = np.arange(0, 5, dt)
    signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*2*t)

    # Initialize DMD
    window_size = 30
    dmd = SvdOnlineDmd(window_size=window_size, max_rank=6, forgetting_factor=0.999)

    # Train on middle portion
    mid_point = len(signal) // 2
    train_start = mid_point - 100
    train_end = mid_point + 100
    dmd.initialize(signal[train_start:train_end])

    # Test backward reconstruction
    backward_steps = mid_point
    x_init = signal[mid_point-1:mid_point+window_size-1]
    # x_init = None
    backward_recon, backward_modes = dmd.reconstruct_backward(backward_steps, x_init)

    # Test forward reconstruction
    forward_steps = len(signal) - mid_point
    x_init = signal[mid_point:mid_point+window_size]
    forward_recon, forward_modes = dmd.decompose_by_modes(forward_steps, x_init)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backward and Forward Reconstruction Test', fontsize=16)

    # Original signal with training region
    ax1 = axes[0, 0]
    ax1.plot(t, signal, 'b-', alpha=0.7, label='Original Signal')
    ax1.axvspan(t[train_start], t[train_end], alpha=0.3, color='green', label='Training Region')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal and Training Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Backward reconstruction
    ax2 = axes[0, 1]
    t_backward = t[mid_point-backward_steps:mid_point]
    t_original_back = t[mid_point-backward_steps:mid_point]
    signal_original_back = signal[mid_point-backward_steps:mid_point]

    ax2.plot(t_original_back, signal_original_back, 'b-', alpha=0.7, label='Original')
    ax2.plot(t_backward, np.real(backward_recon), 'r--', linewidth=2, label='DMD Backward')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Backward Reconstruction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Forward reconstruction
    ax3 = axes[1, 0]
    t_forward = t[mid_point] + np.arange(forward_steps) * dt
    t_original_forward = t[mid_point:mid_point+forward_steps]
    signal_original_forward = signal[mid_point:mid_point+forward_steps]

    ax3.plot(t_original_forward, signal_original_forward, 'b-', alpha=0.7, label='Original')
    ax3.plot(t_forward, np.real(forward_recon), 'r--', linewidth=2, label='DMD Forward')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Forward Reconstruction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Reconstruction errors
    ax4 = axes[1, 1]
    backward_error = np.abs(signal_original_back - np.real(backward_recon))
    forward_error = np.abs(signal_original_forward - np.real(forward_recon))

    ax4.plot(t_backward, backward_error, 'r-', label='Backward Error')
    ax4.plot(t_forward, forward_error, 'g-', label='Forward Error')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Reconstruction Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.show()

    # Print error statistics
    print("=== Backward Reconstruction Test ===")
    print(f"Backward reconstruction error (mean): {np.mean(backward_error):.6f}")
    print(f"Backward reconstruction error (max):  {np.max(backward_error):.6f}")
    print(f"Forward reconstruction error (mean):  {np.mean(forward_error):.6f}")
    print(f"Forward reconstruction error (max):   {np.max(forward_error):.6f}")


if __name__ == "__main__":
    test_backward_reconstruction()
