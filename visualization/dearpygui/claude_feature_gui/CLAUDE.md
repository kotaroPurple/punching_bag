# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GUI application for analyzing load sensor time-series data and extracting machine learning features using DearPyGUI. The application consists of a single Python file with a two-class architecture:

- `LoadSensorAnalyzer`: Core data processing and feature extraction engine
- `LoadSensorGUI`: DearPyGUI-based user interface that wraps the analyzer

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python load_sensor_gui.py
```

## Architecture

### Core Components

**LoadSensorAnalyzer Class:**
- Handles CSV/Excel file loading via pandas
- Implements 20+ feature extraction algorithms across three domains:
  - Statistical features (mean, std, skewness, kurtosis, etc.)
  - Frequency domain features (FFT-based spectral analysis)
  - Time domain features (rolling statistics, change point detection)
- Manages feature selection state for visualization

**LoadSensorGUI Class:**
- Uses DearPyGUI for the interface with three main sections:
  - File loading and column selection controls
  - Feature list (checkboxes) and values display
  - Visualization area for plots
- Integrates matplotlib with DearPyGUI textures for plot rendering
- Implements file dialog, data loading, feature calculation, and export workflows

### Key Technical Details

**Feature Calculation:**
- Operates on single data columns from loaded DataFrames
- Uses configurable window sizes for rolling statistics
- Converts pandas data to numpy arrays for numerical processing
- All features stored in dictionary format for easy serialization

**Visualization Pipeline:**
- matplotlib figures → FigureCanvasAgg → RGBA buffer → DearPyGUI texture
- Supports time-series plots, feature distributions, and correlation matrices
- Plot textures are dynamically created and managed in DearPyGUI's texture registry

**Data Flow:**
File → pandas DataFrame → feature extraction → visualization/export

## Expected Data Format

- CSV or Excel files with headers
- Time-series load sensor data in columns
- User selects specific column for analysis via GUI dropdown