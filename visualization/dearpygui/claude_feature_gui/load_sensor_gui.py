import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from typing import Dict, Any, List, Optional

class LoadSensorAnalyzer:
    def __init__(self) -> None:
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[Dict[str, List[float]]] = None
        self.selected_features: List[str] = []

    def load_data(self, file_path: str) -> bool:
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                return False
            return True
        except Exception as e:
            print(f"Data loading error: {e}")
            return False

    def calculate_features(self, data_column: str, window_size: int = 100) -> Dict[str, List[float]]:
        if self.data is None or data_column not in self.data.columns:
            return {}

        signal_data = np.array(self.data[data_column].values)
        data_length = len(signal_data)

        if data_length < window_size:
            return {}

        # Calculate number of windows and initialize feature arrays
        num_windows = data_length - window_size + 1
        features: Dict[str, List[float]] = {}

        # Initialize arrays for each feature
        feature_names: List[str] = [
            'mean', 'std', 'max', 'min', 'range', 'rms', 'peak_to_peak', 
            'crest_factor', 'skewness', 'kurtosis', 'spectral_centroid', 
            'spectral_energy', 'mean_diff', 'std_diff', 'max_diff', 'zero_crossing_rate'
        ]

        for name in feature_names:
            features[name] = []

        # Calculate features for each sliding window
        for i in range(num_windows):
            window_data = signal_data[i:i+window_size]

            # Statistical features
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            max_val = np.max(window_data)
            min_val = np.min(window_data)
            range_val = max_val - min_val
            rms_val = np.sqrt(np.mean(window_data**2))
            peak_to_peak_val = np.ptp(window_data)
            crest_factor_val = max_val / rms_val if rms_val != 0 else 0
            skewness_val = skew(window_data)
            kurtosis_val = kurtosis(window_data)

            features['mean'].append(mean_val)
            features['std'].append(std_val)
            features['max'].append(max_val)
            features['min'].append(min_val)
            features['range'].append(range_val)
            features['rms'].append(rms_val)
            features['peak_to_peak'].append(peak_to_peak_val)
            features['crest_factor'].append(crest_factor_val)
            features['skewness'].append(skewness_val)
            features['kurtosis'].append(kurtosis_val)

            # Frequency domain features
            fft_data = np.fft.fft(window_data)
            frequencies = np.fft.fftfreq(len(window_data))
            power_spectrum = np.abs(fft_data)**2

            pos_freq_mask = frequencies >= 0
            pos_frequencies = frequencies[pos_freq_mask]
            pos_power = power_spectrum[pos_freq_mask]

            if np.sum(pos_power) > 0:
                spectral_centroid_val = np.sum(pos_frequencies * pos_power) / np.sum(pos_power)
            else:
                spectral_centroid_val = 0
            spectral_energy_val = np.sum(power_spectrum)

            features['spectral_centroid'].append(spectral_centroid_val)
            features['spectral_energy'].append(spectral_energy_val)

            # Change point detection features
            diff_signal = np.diff(window_data)
            mean_diff_val = np.mean(diff_signal)
            std_diff_val = np.std(diff_signal)
            max_diff_val = np.max(np.abs(diff_signal)) if len(diff_signal) > 0 else 0

            features['mean_diff'].append(mean_diff_val)
            features['std_diff'].append(std_diff_val)
            features['max_diff'].append(max_diff_val)

            # Zero crossing
            zero_crossings = np.where(np.diff(np.signbit(window_data)))[0]
            zero_crossing_rate_val = len(zero_crossings) / len(window_data)
            features['zero_crossing_rate'].append(zero_crossing_rate_val)

        self.features = features
        return features

class LoadSensorGUI:
    def __init__(self) -> None:
        self.analyzer: LoadSensorAnalyzer = LoadSensorAnalyzer()
        self.current_file: str = ""

    def setup_gui(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Load Sensor Feature Visualization Tool", width=1200, height=800)

        with dpg.window(label="Main Window", tag="main_window"):
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

            # Feature selection and display
            with dpg.group(horizontal=True):
                # Left: Feature list
                with dpg.child_window(width=250, height=500):
                    dpg.add_text("Feature List")
                    dpg.add_separator()
                    with dpg.group(tag="features_list"):
                        pass

                # Right: Feature values display
                with dpg.child_window(width=250, height=500):
                    dpg.add_text("Feature Values")
                    dpg.add_separator()
                    with dpg.group(tag="features_values"):
                        pass

                # Visualization area (expanded)
                with dpg.child_window(width=680, height=500):
                    dpg.add_text("Visualization")
                    dpg.add_separator()
                    with dpg.plot(label="Plot", height=450, width=660, tag="main_plot"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="y_axis")

            dpg.add_separator()

            # Visualization options
            with dpg.group(horizontal=True):
                dpg.add_button(label="Sensor Data", callback=self.plot_time_series)
                dpg.add_button(label="Features", callback=self.plot_feature_distribution)
                dpg.add_button(label="Combined Plot", callback=self.plot_combined)
                dpg.add_button(label="Save Features", callback=self.save_features)

        # File dialog
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.file_dialog_callback,
                tag="file_dialog", width=700, height=400):
            # dpg.add_file_extension(".*")
            dpg.add_file_extension(".csv", color=(0, 255, 0, 255))
            dpg.add_file_extension(".xlsx", color=(0, 255, 255, 255))

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def select_file(self) -> None:
        dpg.show_item("file_dialog")

    def file_dialog_callback(self, sender: Any, app_data: Dict[str, Any]) -> None:
        self.current_file = app_data['file_path_name']
        dpg.set_value("file_path", self.current_file)

    def load_data(self) -> None:
        if not self.current_file:
            return

        if self.analyzer.load_data(self.current_file) and self.analyzer.data is not None:
            # Add data columns to combo box
            columns = list(self.analyzer.data.columns)
            dpg.configure_item("data_column", items=columns)
            if columns:
                dpg.set_value("data_column", columns[0])

            # Display data information
            info_text = f"Data loaded successfully\nRows: {len(self.analyzer.data)}\nColumns: {len(self.analyzer.data.columns)}"
            try:
                dpg.delete_item("data_info")
            except SystemError:
                pass
            with dpg.group(tag="data_info", parent="features_values"):
                dpg.add_text(info_text)
        else:
            try:
                dpg.delete_item("data_info")
            except SystemError:
                pass
            with dpg.group(tag="data_info", parent="features_values"):
                dpg.add_text("Failed to load data")

    def column_selected(self, sender: Optional[Any] = None, app_data: Optional[Any] = None) -> None:
        pass

    def calculate_features(self) -> None:
        column = dpg.get_value("data_column")
        window_size = dpg.get_value("window_size")

        if not column or self.analyzer.data is None:
            return

        features = self.analyzer.calculate_features(column, window_size)

        # Update feature list
        dpg.delete_item("features_list", children_only=True)
        dpg.delete_item("features_values", children_only=True)

        for feature_name, feature_values in features.items():
            with dpg.group(parent="features_list"):
                dpg.add_checkbox(label=feature_name, tag=f"check_{feature_name}",
                                callback=lambda: self.update_selected_features())

            # Show summary statistics for each feature time series
            if isinstance(feature_values, list) and len(feature_values) > 0:
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                with dpg.group(parent="features_values"):
                    dpg.add_text(f"{feature_name}: μ={mean_val:.4f}, σ={std_val:.4f} (n={len(feature_values)})")
            else:
                with dpg.group(parent="features_values"):
                    dpg.add_text(f"{feature_name}: No data")

    def update_selected_features(self) -> None:
        self.analyzer.selected_features = []
        if self.analyzer.features:
            for feature_name in self.analyzer.features.keys():
                if dpg.get_value(f"check_{feature_name}"):
                    self.analyzer.selected_features.append(feature_name)

    def clear_plot(self) -> None:
        # Clear existing plot series from both y-axes
        for axis_tag in ["y_axis", "y_axis_2"]:
            try:
                children = dpg.get_item_children(axis_tag, 1)
                if children:
                    for child in children:
                        dpg.delete_item(child)
            except SystemError:
                pass

        # Remove second y-axis if it exists
        try:
            dpg.delete_item("y_axis_2")
        except SystemError:
            pass

    def plot_time_series(self) -> None:
        if self.analyzer.data is None:
            return

        column = dpg.get_value("data_column")
        if not column:
            return

        self.clear_plot()

        # Plot original sensor data
        data_values = self.analyzer.data[column].values
        x_values = list(range(len(data_values)))
        y_values = list(data_values)

        dpg.add_line_series(x_values, y_values, label=f"Sensor Data: {column}", parent="y_axis")

        # Update axis labels
        dpg.set_item_label("x_axis", "Time (Sample)")
        dpg.set_item_label("y_axis", "Value")

        # Fit axes to data
        dpg.fit_axis_data("x_axis")
        dpg.fit_axis_data("y_axis")

    def plot_feature_distribution(self) -> None:
        if not self.analyzer.features:
            return

        self.update_selected_features()
        if not self.analyzer.selected_features:
            return

        self.clear_plot()

        # Get window size for x-axis offset
        window_size = dpg.get_value("window_size")

        # Plot selected features as time series
        for i, feature_name in enumerate(self.analyzer.selected_features[:4]):  # Show up to 4 features
            if feature_name in self.analyzer.features:
                feature_values = self.analyzer.features[feature_name]
                if isinstance(feature_values, list) and len(feature_values) > 0:
                    # X values start from window_size//2 to center the feature values
                    x_values = list(range(window_size//2, window_size//2 + len(feature_values)))
                    dpg.add_line_series(x_values, feature_values,
                                        label=f"Feature: {feature_name}", parent="y_axis")

        # Update axis labels
        dpg.set_item_label("x_axis", "Time (Sample)")
        dpg.set_item_label("y_axis", "Feature Values")

        # Fit axes to data
        dpg.fit_axis_data("x_axis")
        dpg.fit_axis_data("y_axis")

    def plot_combined(self) -> None:
        if self.analyzer.data is None or not self.analyzer.features:
            return

        column = dpg.get_value("data_column")
        if not column:
            return

        self.update_selected_features()
        if not self.analyzer.selected_features:
            return

        self.clear_plot()

        # Add second y-axis for features
        dpg.add_plot_axis(dpg.mvYAxis, label="Feature Values", tag="y_axis_2")

        # Plot original sensor data on primary y-axis
        data_values = self.analyzer.data[column].values
        x_values = list(range(len(data_values)))
        y_values = list(data_values)

        dpg.add_line_series(x_values, y_values, label=f"Sensor: {column}", parent="y_axis")

        # Get window size for feature x-axis offset
        window_size = dpg.get_value("window_size")

        # Plot selected features on secondary y-axis
        for feature_name in self.analyzer.selected_features[:3]:  # Show up to 3 features
            if feature_name in self.analyzer.features:
                feature_values = self.analyzer.features[feature_name]
                if isinstance(feature_values, list) and len(feature_values) > 0:
                    # X values start from window_size//2 to center the feature values
                    feature_x_values = list(range(window_size//2, window_size//2 + len(feature_values)))
                    dpg.add_line_series(feature_x_values, feature_values,
                                        label=f"Feature: {feature_name}", parent="y_axis_2")

        # Update axis labels
        dpg.set_item_label("x_axis", "Time (Sample)")
        dpg.set_item_label("y_axis", f"Sensor: {column}")

        # Fit axes to data
        dpg.fit_axis_data("x_axis")
        dpg.fit_axis_data("y_axis")
        dpg.fit_axis_data("y_axis_2")

    def save_features(self) -> None:
        if not self.analyzer.features:
            return

        feature_df = pd.DataFrame([self.analyzer.features])
        output_file = self.current_file.replace('.csv', '_features.csv').replace('.xlsx', '_features.csv')
        feature_df.to_csv(output_file, index=False)

        # Show save confirmation in the feature values area
        try:
            dpg.delete_item("save_info")
        except SystemError:
            pass
        with dpg.group(tag="save_info", parent="features_values"):
            dpg.add_text(f"Features saved to: {output_file}")

    def run(self) -> None:
        self.setup_gui()
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    gui = LoadSensorGUI()
    gui.run()
