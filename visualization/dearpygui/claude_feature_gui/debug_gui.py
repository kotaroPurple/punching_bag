#!/usr/bin/env python3
"""
Debug version of the GUI with additional logging and error handling
"""

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import traceback

class DebugLoadSensorAnalyzer:
    def __init__(self):
        self.data = None
        self.features = None
        self.selected_features = []
        
    def load_data(self, file_path: str) -> bool:
        try:
            print(f"Loading file: {file_path}")
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                return False
            
            print(f"Data loaded successfully: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Data loading error: {e}")
            traceback.print_exc()
            return False
    
    def calculate_features(self, data_column: str, window_size: int = 100) -> Dict[str, Any]:
        try:
            print(f"Calculating features for column: {data_column}")
            if self.data is None:
                print("No data loaded!")
                return {}
            
            if data_column not in self.data.columns:
                print(f"Column '{data_column}' not found in data!")
                print(f"Available columns: {list(self.data.columns)}")
                return {}
            
            signal_data = np.array(self.data[data_column].values)
            print(f"Signal data shape: {signal_data.shape}")
            features = {}
            
            # Statistical features
            features['mean'] = np.mean(signal_data)
            features['std'] = np.std(signal_data)
            features['max'] = np.max(signal_data)
            features['min'] = np.min(signal_data)
            features['range'] = features['max'] - features['min']
            features['rms'] = np.sqrt(np.mean(signal_data**2))
            features['peak_to_peak'] = np.ptp(signal_data)
            features['crest_factor'] = features['max'] / features['rms'] if features['rms'] != 0 else 0
            features['skewness'] = skew(signal_data)
            features['kurtosis'] = kurtosis(signal_data)
            
            # Frequency domain features
            fft_data = np.fft.fft(signal_data)
            frequencies = np.fft.fftfreq(len(signal_data))
            power_spectrum = np.abs(fft_data)**2
            
            positive_freqs = frequencies[:len(frequencies)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            if np.sum(positive_power) != 0:
                features['spectral_centroid'] = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
            else:
                features['spectral_centroid'] = 0
            features['spectral_energy'] = np.sum(power_spectrum)
            
            # Time domain features (rolling statistics)
            rolling_mean = pd.Series(signal_data).rolling(window=window_size).mean()
            rolling_std = pd.Series(signal_data).rolling(window=window_size).std()
            
            features['mean_of_rolling_mean'] = np.nanmean(rolling_mean)
            features['std_of_rolling_mean'] = np.nanstd(rolling_mean)
            features['mean_of_rolling_std'] = np.nanmean(rolling_std)
            features['std_of_rolling_std'] = np.nanstd(rolling_std)
            
            # Change point detection features
            diff_signal = np.diff(signal_data)
            features['mean_diff'] = np.mean(diff_signal)
            features['std_diff'] = np.std(diff_signal)
            features['max_diff'] = np.max(np.abs(diff_signal))
            
            # Zero crossing
            zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
            features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
            
            self.features = features
            print(f"Calculated {len(features)} features successfully")
            return features
            
        except Exception as e:
            print(f"Feature calculation error: {e}")
            traceback.print_exc()
            return {}

class DebugLoadSensorGUI:
    def __init__(self):
        self.analyzer = DebugLoadSensorAnalyzer()
        self.current_file = ""
        self.plot_texture = None
        
    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Debug Load Sensor GUI", width=1200, height=800)
        
        with dpg.window(label="Debug Main Window", tag="main_window"):
            # File loading section
            with dpg.group(horizontal=True):
                dpg.add_text("Data File:")
                dpg.add_input_text(tag="file_path", width=300, readonly=True)
                dpg.add_button(label="Select File", callback=self.select_file)
                dpg.add_button(label="Load Data", callback=self.load_data)
            
            dpg.add_separator()
            
            # Data column selection
            with dpg.group(horizontal=True):
                dpg.add_text("Data Column:")
                dpg.add_combo(tag="data_column", width=200, callback=self.column_selected)
                dpg.add_text("Window Size:")
                dpg.add_input_int(tag="window_size", default_value=100, width=100)
                dpg.add_button(label="Calculate Features", callback=self.calculate_features)
            
            dpg.add_separator()
            
            # Feature display
            with dpg.group(horizontal=True):
                with dpg.child_window(width=400, height=400):
                    dpg.add_text("Feature Values")
                    dpg.add_separator()
                    with dpg.group(tag="features_display"):
                        pass
                
                with dpg.child_window(width=600, height=400):
                    dpg.add_text("Debug Log")
                    dpg.add_separator()
                    with dpg.group(tag="debug_log"):
                        pass
            
            dpg.add_separator()
            
            # Test buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="Test Load Sine Wave", callback=self.test_load_sine_wave)
                dpg.add_button(label="Test Load Step Response", callback=self.test_load_step_response)
                dpg.add_button(label="Clear Log", callback=self.clear_log)
        
        # File dialog
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.file_dialog_callback,
                           tag="file_dialog", width=700, height=400):
            # dpg.add_file_extension(".*")
            dpg.add_file_extension(".csv", color=(0, 255, 0, 255))
            dpg.add_file_extension(".xlsx", color=(0, 255, 255, 255))
        
        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def log_message(self, message: str):
        """Add message to debug log"""
        with dpg.group(parent="debug_log"):
            dpg.add_text(message)
        print(message)  # Also print to console
    
    def clear_log(self):
        """Clear debug log"""
        dpg.delete_item("debug_log", children_only=True)
    
    def select_file(self):
        self.log_message("Opening file dialog...")
        dpg.show_item("file_dialog")
    
    def file_dialog_callback(self, sender, app_data):
        self.current_file = app_data['file_path_name']
        dpg.set_value("file_path", self.current_file)
        self.log_message(f"Selected file: {self.current_file}")
    
    def load_data(self):
        if not self.current_file:
            self.log_message("No file selected!")
            return
        
        self.log_message(f"Loading data from: {self.current_file}")
        
        if self.analyzer.load_data(self.current_file) and self.analyzer.data is not None:
            # Add data columns to combo box
            columns = list(self.analyzer.data.columns)
            dpg.configure_item("data_column", items=columns)
            if columns:
                dpg.set_value("data_column", columns[0])
                self.log_message(f"Data columns available: {columns}")
            
            self.log_message(f"Data loaded: {self.analyzer.data.shape[0]} rows, {self.analyzer.data.shape[1]} columns")
        else:
            self.log_message("Failed to load data!")
    
    def column_selected(self, sender=None, app_data=None):
        column = dpg.get_value("data_column")
        self.log_message(f"Selected column: {column}")
    
    def calculate_features(self):
        column = dpg.get_value("data_column")
        window_size = dpg.get_value("window_size")
        
        if not column or self.analyzer.data is None:
            self.log_message("No column selected or no data loaded!")
            return
        
        self.log_message(f"Calculating features for column '{column}' with window size {window_size}")
        
        features = self.analyzer.calculate_features(column, window_size)
        
        if features:
            # Update feature display
            dpg.delete_item("features_display", children_only=True)
            
            for feature_name, feature_value in features.items():
                with dpg.group(parent="features_display"):
                    dpg.add_text(f"{feature_name}: {feature_value:.4f}")
            
            self.log_message(f"Successfully calculated {len(features)} features")
        else:
            self.log_message("Feature calculation failed!")
    
    def test_load_sine_wave(self):
        """Test loading sine wave data directly"""
        self.current_file = "test_data_sine_wave.csv"
        dpg.set_value("file_path", self.current_file)
        self.load_data()
    
    def test_load_step_response(self):
        """Test loading step response data directly"""
        self.current_file = "test_data_step_response.csv"
        dpg.set_value("file_path", self.current_file)
        self.load_data()
    
    def run(self):
        self.setup_gui()
        self.log_message("Debug GUI started")
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    print("Starting Debug GUI...")
    gui = DebugLoadSensorGUI()
    gui.run()