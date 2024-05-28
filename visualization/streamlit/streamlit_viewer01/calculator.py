
import time
from dataclasses import dataclass

@dataclass
class ComputationParameters:
    value: int = 0
    mode: str = ''


class ComputationManager:
    # @st.cache_data
    def expensive_computation(self, param: ComputationParameters):
        time.sleep(2)  # Simulate a long computation
        x = param.value
        match param.mode:
            case "Option 1":
                y = x
            case "Option 2":
                y = pow(x, 2)
            case "Option 3":
                y = pow(x, 3)
            case _:
                y = 0
        return y

    # @st.cache_data
    def load_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return data
