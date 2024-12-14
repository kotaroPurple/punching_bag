
# ref: https://discuss.streamlit.io/t/cells-rendered-as-text-rather-than-html-links/48272

import numpy as np
import pandas as pd
# import streamlit as st
import matplotlib.pyplot as plt
import base64
import io
from matplotlib.figure import Figure

from st_aggrid import AgGrid, JsCode, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder



def plot_to_base64(fig: Figure) -> str:
    """
    Matplotlib の Figure オブジェクトを Base64 エンコードされた画像データ URI に変換します。
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


n_rows = 5
a_list: list[str] = ['A', 'B', 'C']
b_list: list[int] = [0, 10, 20]

a_values: list[str] = []
b_values: list[int] = []
plot_values: list[str] = []

for n in range(n_rows):
    # condition
    a = np.random.choice(a_list)
    b = np.random.choice(b_list)
    # plot
    x = np.arange(10)
    y = np.random.choice(10, len(x))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    data_uri = plot_to_base64(fig)
    # append
    a_values.append(a)
    b_values.append(b)
    plot_values.append(data_uri)

df = pd.DataFrame({
    'a': a_values,
    'b': b_values,
    'plot': plot_values
})

cell_renderer = JsCode("""
class ImageCellRenderer {
    init(params) {
        this.eGui = document.createElement('div');
        if (params.value) {
            const img = document.createElement('img');
            img.src = params.value;
            //img.style.width = '100px'; // 画像の幅を設定
            //img.style.height = 'auto'; // 高さを自動調整
            img.style.height = '100px';
            this.eGui.appendChild(img);
        } else {
            this.eGui.innerText = '';
        }
    }
    getGui() {
        return this.eGui;
    }
}
""")



# GridOptionsBuilder のインスタンスを作成
gb = GridOptionsBuilder.from_dataframe(df)

# デフォルトの行の高さを設定
gb.configure_grid_options(rowHeight=100)

# '画像URL' 列にカスタムセルレンダラーを設定
gb.configure_column(
    "plot",
    header_name="plot",
    cellRenderer=cell_renderer,
)

# gb.configure_auto_height(True)


# グリッドオプションをビルド
gridOptions = gb.build()

# AgGrid の表示
AgGrid(
    df,
    gridOptions=gridOptions,
    enable_enterprise_modules=False,
    allow_unsafe_jscode=True,
    height=1000,
    # width='100%',
    width=500,
    fit_columns_on_grid_load=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
)
