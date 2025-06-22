#!/usr/bin/env python3
"""
Test script to debug GUI data loading issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_sensor_gui import LoadSensorAnalyzer
import pandas as pd

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading functionality...")
    
    analyzer = LoadSensorAnalyzer()
    
    # Test loading the generated CSV file
    test_file = "test_data_sine_wave.csv"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        return False
    
    print(f"Loading file: {test_file}")
    
    # Load data
    success = analyzer.load_data(test_file)
    
    if not success:
        print("Failed to load data!")
        return False
    
    print("Data loaded successfully!")
    print(f"Data shape: {analyzer.data.shape}")
    print(f"Columns: {list(analyzer.data.columns)}")
    print(f"First few rows:")
    print(analyzer.data.head())
    
    # Test feature calculation
    print("\nTesting feature calculation...")
    data_column = "load_sensor"
    
    if data_column not in analyzer.data.columns:
        print(f"Error: Column '{data_column}' not found in data!")
        return False
    
    features = analyzer.calculate_features(data_column, window_size=100)
    
    if not features:
        print("No features calculated!")
        return False
    
    print(f"Calculated {len(features)} features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    return True

def test_all_datasets():
    """Test loading all generated datasets"""
    test_files = [
        "test_data_sine_wave.csv",
        "test_data_step_response.csv", 
        "test_data_impulse_response.csv",
        "test_data_complex_loading.csv",
        "test_data_high_frequency.csv"
    ]
    
    analyzer = LoadSensorAnalyzer()
    
    for test_file in test_files:
        print(f"\n{'='*50}")
        print(f"Testing: {test_file}")
        print(f"{'='*50}")
        
        if not os.path.exists(test_file):
            print(f"File not found: {test_file}")
            continue
        
        success = analyzer.load_data(test_file)
        if not success:
            print(f"Failed to load: {test_file}")
            continue
        
        print(f"Shape: {analyzer.data.shape}")
        print(f"Columns: {list(analyzer.data.columns)}")
        
        # Test feature calculation on load_sensor column
        if "load_sensor" in analyzer.data.columns:
            features = analyzer.calculate_features("load_sensor", 100)
            print(f"Features calculated: {len(features)}")
            
            # Show first 5 features
            feature_items = list(features.items())[:5]
            for name, value in feature_items:
                print(f"  {name}: {value:.4f}")
        else:
            print("No 'load_sensor' column found")

if __name__ == "__main__":
    print("GUI Data Loading Test")
    print("=" * 30)
    
    # Test individual file
    if test_data_loading():
        print("\n✓ Basic data loading test passed!")
    else:
        print("\n✗ Basic data loading test failed!")
    
    # Test all datasets
    print("\n" + "=" * 50)
    print("Testing all datasets...")
    test_all_datasets()
    
    print("\nTest completed!")