# from shiny.express import input, render, ui

# ui.input_selectize(
#     "var", "Select variable",
#     choices=["bill_length_mm", "body_mass_g"]
# )

# @render.plot
# def hist():
#     from matplotlib import pyplot as plt
#     from palmerpenguins import load_penguins

#     df = load_penguins()
#     df[input.var()].hist(grid=False)
#     plt.xlabel(input.var())
#     plt.ylabel("count")

# app.py

import pathlib
from shiny import App, module, ui, render, reactive

## Page A

@module.ui
def a_ui():

    return ui.TagList(
        ui.input_numeric("num", "Number for Panel A", value=0),
        ui.output_text_verbatim("x")
    )

@module.server
def a_server(input, output, session):

    @output
    @render.text
    def x():
        n = input.num()
        return f"{n} times 2 = { n * 2 }"

## Page B

@module.ui
def b_ui():
    return ui.TagList(
        ui.input_numeric("num", "Number for Panel B", value=0),
        ui.output_text_verbatim("x")
    )

@module.server
def b_server(input, output, session):
    @output
    @render.text
    def x():
        n = input.num()
        return f"{n} times 3 = {n * 3}"

def app_ui(request):

    _ui = ui.page_fixed(
        ui.head_content(
            ui.tags.script(src="js/msg.js")                 # (1)
        ),
        ui.panel_title("Main Title"),
        ui.navset_tab_card(
            ui.nav("Page A", a_ui("pageA"), value="a"),     # (2a)
            ui.nav("Page B", b_ui("pageB"), value="b"),     # (2b)
            id="page",                                      # (3)
            selected=request.query_params.get("page", "a")  # (4)
        )
    )

    return _ui

def app_server(input, output, session):

    a_server("pageA")
    b_server("pageB")

    @reactive.Effect
    @reactive.event(input.page)                                # (5)
    async def _():
        msg = { "page": input.page() }                         # (6)
        await session.send_custom_message("replaceState", msg) # (7)


www_dir = pathlib.Path(__file__).parent / "www"                # (8)
app = App(app_ui, app_server, static_assets=www_dir)
