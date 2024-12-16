
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# タイトルの設定
st.title("2x2 サブプロットでの2本の曲線アニメーション例（Plotly内スライダーとフレーム番号表示）")

# アニメーションのパラメータ設定
num_frames = 60  # フレーム数
x = np.linspace(0, 4 * np.pi, 200)  # x軸のデータポイント

# 色の定義
colors = ['blue', 'red', 'green', 'orange']

# フレームデータのプリコンピュート
frames_data = []
for frame_num in range(num_frames):
    frame_data = []
    for i in range(1, 5):
        phase = frame_num * (2 * np.pi / num_frames) + i  # 各サブプロットに異なる位相を適用
        y1 = np.sin(x + phase)
        y2 = np.cos(x + phase)
        frame_data.append((y1, y2))
    frames_data.append(frame_data)

# サブプロットの作成（2行2列）
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "sub plot 1: sin 波 & cos 波",
        "sub plot 2: sin 波 & cos 波",
        "sub plot 3: sin 波 & cos 波",
        "sub plot 4: sin 波 & cos 波"
    )
)

# 初期フレームのトレース追加
for i in range(1, 5):
    y1, y2 = frames_data[0][i-1]
    # サイン波トレース
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            mode='lines',
            name=f'sin 波 {i}',
            line=dict(color=colors[i-1])
        ),
        row=(i-1)//2 + 1, col=(i-1)%2 + 1
    )
    # コサイン波トレース
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            mode='lines',
            name=f'cos 波 {i}',
            line=dict(color=colors[i-1], dash='dash')
        ),
        row=(i-1)//2 + 1, col=(i-1)%2 + 1
    )

# フレームの作成
frames = []
for frame_num in range(num_frames):
    frame_data = []
    for i in range(1, 5):
        y1, y2 = frames_data[frame_num][i-1]
        # サイン波トレースの更新
        frame_data.append(
            go.Scatter(
                x=x,
                y=y1,
                mode='lines',
                name=f'sin 波 {i}',
                line=dict(color=colors[i-1])
            )
        )
        # コサイン波トレースの更新
        frame_data.append(
            go.Scatter(
                x=x,
                y=y2,
                mode='lines',
                name=f'cos 波 {i}',
                line=dict(color=colors[i-1], dash='dash')
            )
        )
    # フレーム番号のアノテーションを追加
    frame = go.Frame(
        data=frame_data,
        name=str(frame_num),
        # layout=go.Layout(
        #     annotations=[
        #         dict(
        #             text=f"フレーム: {frame_num}",
        #             x=0.5, y=1.05,
        #             xref="paper", yref="paper",
        #             showarrow=False,
        #             font=dict(size=14)
        #         )
        #     ]
        # )
    )
    frames.append(frame)

fig.frames = frames

# アニメーションのレイアウト設定
fig.update_layout(
    title="2x2 サブプロットでの2本の曲線アニメーション",
    height=800,
    showlegend=False,  # 凡例を表示したい場合はTrueに変更
    margin=dict(l=50, r=50, t=100, b=50),
    updatemenus=[
        dict(
            type="buttons",
            direction="left",  # ボタンを横並びに配置
            showactive=False,
            x=0.0,  # スライダーの左側に配置
            y=0.0,  # スライダーの近くに配置
            xanchor="right",
            yanchor="top",
            pad=dict(t=50, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode='immediate'
                        )
                    ]
                ),
                dict(
                    label="Stop",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": "Ratio: "},
            pad={"t": 50},
            x=0.1,
            y=0.02,
            len=0.8,
            steps=[
                dict(
                    args=[
                        [str(frame_num)],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )
                    ],
                    label=f'{frame_num / 100:.2f}',
                    method="animate"
                ) for frame_num in range(num_frames)
            ]
        )
    ]
)

# 各サブプロットの軸ラベル設定
for i in range(1, 5):
    fig.update_xaxes(title_text="X軸", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    fig.update_yaxes(title_text="Y軸", row=(i-1)//2 + 1, col=(i-1)%2 + 1)

# # アノテーションの初期フレーム番号表示
# fig.update_layout(
#     annotations=[
#         dict(
#             text="フレーム: 0",
#             x=0.5, y=1.05,
#             xref="paper", yref="paper",
#             showarrow=False,
#             font=dict(size=14)
#         )
#     ]
# )

# グラフの表示
st.plotly_chart(fig, use_container_width=True)
