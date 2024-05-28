
import streamlit as st

from calculator import ComputationManager
from calculator import ComputationParameters


class StateManager:
    def __init__(self):
        if 'selected_option' not in st.session_state:
            st.session_state.selected_option = "Option 1"
        if 'slider_value' not in st.session_state:
            st.session_state.slider_value = 50
        if 'last_selected_option' not in st.session_state:
            st.session_state.last_selected_option = st.session_state.selected_option
        if 'last_slider_value' not in st.session_state:
            st.session_state.last_slider_value = st.session_state.slider_value
        if 'computation_needed' not in st.session_state:
            st.session_state.computation_needed = False
        if 'computation_manager' not in st.session_state:
            st.session_state.computation_manager = ComputationManager()

    @property
    def selected_option(self):
        return st.session_state.selected_option

    @selected_option.setter
    def selected_option(self, value):
        st.session_state.selected_option = value

    @property
    def slider_value(self):
        return st.session_state.slider_value

    @slider_value.setter
    def slider_value(self, value):
        st.session_state.slider_value = value

    def update_last_values(self):
        st.session_state.last_selected_option = self.selected_option
        st.session_state.last_slider_value = self.slider_value

    def has_changed(self):
        return (st.session_state.selected_option != st.session_state.last_selected_option or
                st.session_state.slider_value != st.session_state.last_slider_value)

    @property
    def computation_needed(self):
        return st.session_state.computation_needed

    @computation_needed.setter
    def computation_needed(self, value):
        st.session_state.computation_needed = value

    def get_computation_manager(self):
        return st.session_state.computation_manager

    def get_computation_parameters(self) -> ComputationParameters:
        param = ComputationParameters(self.slider_value, self.selected_option)
        return param


class StreamlitViewer:
    def __init__(self):
        self.title = "Streamlit Viewer"
        self.sidebar_title = "Sidebar"
        self.sidebar_options = ["Option 1", "Option 2", "Option 3"]
        self.state_manager = StateManager()
        self.computation_manager = self.state_manager.get_computation_manager()

    def display_title(self):
        st.title(self.title)

    def display_sidebar(self):
        st.sidebar.title(self.sidebar_title)
        self.state_manager.selected_option = st.sidebar.selectbox(
            "Choose an option:", self.sidebar_options, index=self.sidebar_options.index(self.state_manager.selected_option)
        )
        st.sidebar.write(f"You selected: {self.state_manager.selected_option}")

        self.state_manager.slider_value = st.sidebar.slider(
            "Select a value", 0, 100, self.state_manager.slider_value
        )
        st.sidebar.write(f"Slider value: {self.state_manager.slider_value}")

    def display_main_content(self):
        st.write("Welcome to the Streamlit Viewer!")
        st.write("This is a simple template for a Streamlit app.")
        st.write("You can customize this according to your needs.")

        file_path = "example.txt"
        try:
            file_content = self.computation_manager.load_file(file_path)
            st.write("File content:")
            st.text(file_content)
        except FileNotFoundError:
            st.error("File not found. Please make sure 'example.txt' exists.")

        if self.state_manager.has_changed():
            self.state_manager.computation_needed = True
            st.write("Values have changed. Please press the button to compute.")

        if st.button("Compute") and self.state_manager.computation_needed:
            param = self.state_manager.get_computation_parameters()
            result = self.computation_manager.expensive_computation(param)
            st.session_state.computation_result = result
            self.state_manager.computation_needed = False
            self.state_manager.update_last_values()
            st.write(f"Computation result: {st.session_state.computation_result}")
        else:
            if 'computation_result' in st.session_state:
                st.write("Computation already done...")
                st.write(f"Computation result: {st.session_state.computation_result}")
            else:
                st.write("No computation yet")

    def run(self):
        self.display_title()
        self.display_sidebar()
        self.display_main_content()


if __name__ == "__main__":
    viewer = StreamlitViewer()
    viewer.run()

