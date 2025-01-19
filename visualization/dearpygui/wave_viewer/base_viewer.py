
# import numpy as np
import dearpygui.dearpygui as dpg
# from numpy.typing import NDArray


# call it before generating tags
dpg.create_context()


IMAGE_TEXTURE_TAG = dpg.generate_uuid()


class BaseViewer:
    def __init__(
            self, title: str, gui_width: int, gui_height: int, interval_sec: float) -> None:
        self._gui_width = gui_width
        self._gui_height = gui_height
        self._interval = interval_sec
        self._elapsed_time = 0.
        # prepare gui
        self._setup_gui(title, self._gui_width, self._gui_height)
        self._setup_window()

    def start(self) -> None:
        dpg.show_viewport()

    def destroy(self) -> None:
        dpg.destroy_context()

    def is_running(self) -> bool:
        return dpg.is_dearpygui_running()

    def render(self) -> None:
        # interval ごとに描画する
        if ((time_from_start := dpg.get_total_time()) - self._elapsed_time) >= self._interval:
            self._elapsed_time = time_from_start
            print(self._elapsed_time)
            self._render_main()
        dpg.render_dearpygui_frame()

    def _setup_gui(self, title: str, width: int, height: int) -> None:
        dpg.create_viewport(title=title, width=width, height=height)
        dpg.setup_dearpygui()

    def _setup_window(self) -> None:
        pass

    def _render_main(self) -> None:
        pass


if __name__ == '__main__':
    gui_width = 400
    gui_height = 600
    gui = BaseViewer('Hello World', gui_width, gui_height, 1.)

    gui.start()

    while gui.is_running():
        gui.render()

    gui.destroy()
