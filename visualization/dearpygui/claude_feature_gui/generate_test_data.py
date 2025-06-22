import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import argparse
import os

class LoadSensorDataGenerator:
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        self.time_step = 1.0 / sampling_rate
    
    def generate_sine_wave(self, duration: float, frequency: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
        """Generate a sine wave signal"""
        t = np.arange(0, duration, self.time_step)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    def generate_step_function(self, duration: float, step_times: List[float], step_values: List[float]) -> np.ndarray:
        """Generate a step function signal"""
        t = np.arange(0, duration, self.time_step)
        signal = np.zeros_like(t)
        
        for i, step_time in enumerate(step_times):
            step_idx = int(step_time * self.sampling_rate)
            if step_idx < len(signal):
                signal[step_idx:] += step_values[i]
        
        return signal
    
    def generate_ramp(self, duration: float, start_value: float, end_value: float, start_time: float = 0.0) -> np.ndarray:
        """Generate a ramp signal"""
        t = np.arange(0, duration, self.time_step)
        signal = np.zeros_like(t)
        
        start_idx = int(start_time * self.sampling_rate)
        if start_idx < len(signal):
            ramp_duration = duration - start_time
            slope = (end_value - start_value) / ramp_duration
            signal[start_idx:] = start_value + slope * (t[start_idx:] - start_time)
        
        return signal
    
    def generate_impulse(self, duration: float, impulse_times: List[float], impulse_amplitudes: List[float], impulse_width: float = 0.01) -> np.ndarray:
        """Generate impulse signals"""
        t = np.arange(0, duration, self.time_step)
        signal = np.zeros_like(t)
        
        for imp_time, imp_amp in zip(impulse_times, impulse_amplitudes):
            imp_idx = int(imp_time * self.sampling_rate)
            width_samples = int(impulse_width * self.sampling_rate)
            
            start_idx = max(0, imp_idx - width_samples // 2)
            end_idx = min(len(signal), imp_idx + width_samples // 2)
            
            if start_idx < end_idx:
                signal[start_idx:end_idx] = imp_amp
        
        return signal
    
    def generate_exponential_decay(self, duration: float, initial_value: float, decay_constant: float, start_time: float = 0.0) -> np.ndarray:
        """Generate exponential decay signal"""
        t = np.arange(0, duration, self.time_step)
        signal = np.zeros_like(t)
        
        start_idx = int(start_time * self.sampling_rate)
        if start_idx < len(signal):
            signal[start_idx:] = initial_value * np.exp(-decay_constant * (t[start_idx:] - start_time))
        
        return signal
    
    def add_noise(self, signal: np.ndarray, noise_level: float = 0.05, noise_type: str = 'gaussian') -> np.ndarray:
        """Add noise to the signal"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level * np.std(signal), noise_level * np.std(signal), signal.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return signal + noise
    
    def generate_composite_signal(self, duration: float, components: List[dict], noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a composite signal from multiple components"""
        t = np.arange(0, duration, self.time_step)
        signal = np.zeros_like(t)
        
        for component in components:
            comp_type = component['type']
            
            if comp_type == 'sine':
                comp_signal = self.generate_sine_wave(duration, component['frequency'], component['amplitude'], component.get('phase', 0.0))
            elif comp_type == 'step':
                comp_signal = self.generate_step_function(duration, component['step_times'], component['step_values'])
            elif comp_type == 'ramp':
                comp_signal = self.generate_ramp(duration, component['start_value'], component['end_value'], component.get('start_time', 0.0))
            elif comp_type == 'impulse':
                comp_signal = self.generate_impulse(duration, component['impulse_times'], component['impulse_amplitudes'], component.get('impulse_width', 0.01))
            elif comp_type == 'exponential_decay':
                comp_signal = self.generate_exponential_decay(duration, component['initial_value'], component['decay_constant'], component.get('start_time', 0.0))
            else:
                continue
            
            signal += comp_signal
        
        if noise_level > 0:
            signal = self.add_noise(signal, noise_level)
        
        return t, signal

def create_test_datasets():
    """Create various test datasets for load sensor analysis"""
    generator = LoadSensorDataGenerator(sampling_rate=1000)
    
    # Dataset 1: Simple sine wave with noise
    print("Generating Dataset 1: Simple sine wave...")
    t1, signal1 = generator.generate_composite_signal(
        duration=10.0,
        components=[
            {'type': 'sine', 'frequency': 2.0, 'amplitude': 100.0},
            {'type': 'sine', 'frequency': 5.0, 'amplitude': 30.0}
        ],
        noise_level=0.05
    )
    
    df1 = pd.DataFrame({
        'time': t1,
        'load_sensor': signal1,
        'temperature': 25.0 + 2.0 * np.sin(2 * np.pi * 0.1 * t1) + np.random.normal(0, 0.5, len(t1))
    })
    df1.to_csv('test_data_sine_wave.csv', index=False)
    
    # Dataset 2: Step response
    print("Generating Dataset 2: Step response...")
    t2, signal2 = generator.generate_composite_signal(
        duration=20.0,
        components=[
            {'type': 'step', 'step_times': [2.0, 8.0, 15.0], 'step_values': [50.0, -30.0, 80.0]},
            {'type': 'exponential_decay', 'initial_value': 20.0, 'decay_constant': 0.5, 'start_time': 5.0}
        ],
        noise_level=0.03
    )
    
    df2 = pd.DataFrame({
        'time': t2,
        'load_sensor': signal2,
        'vibration': 0.1 * np.random.randn(len(t2))
    })
    df2.to_csv('test_data_step_response.csv', index=False)
    
    # Dataset 3: Impulse response
    print("Generating Dataset 3: Impulse response...")
    t3, signal3 = generator.generate_composite_signal(
        duration=15.0,
        components=[
            {'type': 'impulse', 'impulse_times': [2.0, 5.0, 8.0, 12.0], 'impulse_amplitudes': [200.0, -150.0, 300.0, -100.0], 'impulse_width': 0.05},
            {'type': 'sine', 'frequency': 1.0, 'amplitude': 20.0}
        ],
        noise_level=0.08
    )
    
    df3 = pd.DataFrame({
        'time': t3,
        'load_sensor': signal3,
        'acceleration': np.gradient(np.gradient(signal3)) * 1000
    })
    df3.to_csv('test_data_impulse_response.csv', index=False)
    
    # Dataset 4: Complex loading pattern
    print("Generating Dataset 4: Complex loading pattern...")
    t4, signal4 = generator.generate_composite_signal(
        duration=30.0,
        components=[
            {'type': 'ramp', 'start_value': 0.0, 'end_value': 100.0, 'start_time': 0.0},
            {'type': 'sine', 'frequency': 0.5, 'amplitude': 50.0},
            {'type': 'sine', 'frequency': 3.0, 'amplitude': 15.0},
            {'type': 'step', 'step_times': [10.0, 20.0], 'step_values': [30.0, -40.0]},
            {'type': 'impulse', 'impulse_times': [5.0, 15.0, 25.0], 'impulse_amplitudes': [80.0, -60.0, 100.0]}
        ],
        noise_level=0.04
    )
    
    df4 = pd.DataFrame({
        'time': t4,
        'load_sensor': signal4,
        'strain': signal4 * 0.001 + np.random.normal(0, 0.0001, len(signal4)),
        'displacement': np.cumsum(signal4) * 0.0001 + np.random.normal(0, 0.001, len(signal4))
    })
    df4.to_csv('test_data_complex_loading.csv', index=False)
    
    # Dataset 5: High-frequency data
    print("Generating Dataset 5: High-frequency data...")
    generator_hf = LoadSensorDataGenerator(sampling_rate=5000)
    t5, signal5 = generator_hf.generate_composite_signal(
        duration=5.0,
        components=[
            {'type': 'sine', 'frequency': 50.0, 'amplitude': 100.0},
            {'type': 'sine', 'frequency': 120.0, 'amplitude': 30.0},
            {'type': 'sine', 'frequency': 300.0, 'amplitude': 10.0}
        ],
        noise_level=0.1
    )
    
    df5 = pd.DataFrame({
        'time': t5,
        'load_sensor': signal5
    })
    df5.to_csv('test_data_high_frequency.csv', index=False)
    
    print("\nTest datasets created successfully:")
    print("- test_data_sine_wave.csv (10s, 2Hz+5Hz sine waves)")
    print("- test_data_step_response.csv (20s, step functions + exponential decay)")
    print("- test_data_impulse_response.csv (15s, impulse responses)")
    print("- test_data_complex_loading.csv (30s, complex multi-component signal)")
    print("- test_data_high_frequency.csv (5s, high-frequency components)")

def plot_test_data():
    """Plot all generated test datasets"""
    files = [
        'test_data_sine_wave.csv',
        'test_data_step_response.csv', 
        'test_data_impulse_response.csv',
        'test_data_complex_loading.csv',
        'test_data_high_frequency.csv'
    ]
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    for i, file in enumerate(files):
        if os.path.exists(file):
            df = pd.read_csv(file)
            axes[i].plot(df['time'], df['load_sensor'])
            axes[i].set_title(f'Dataset {i+1}: {file}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Load (N)')
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_datasets_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate test data for load sensor analysis')
    parser.add_argument('--plot', action='store_true', help='Plot the generated datasets')
    args = parser.parse_args()
    
    create_test_datasets()
    
    if args.plot:
        plot_test_data()