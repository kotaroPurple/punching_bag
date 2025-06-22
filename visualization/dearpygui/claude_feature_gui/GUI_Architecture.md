# Load Sensor GUI アプリケーション構成

## 概要

時系列ロードセンサーデータの分析と機械学習特徴量抽出を行うDearPyGUIベースのGUIアプリケーションです。

## アーキテクチャ

```mermaid
graph TB
    subgraph "Application Layer"
        A[LoadSensorGUI]
        B[LoadSensorAnalyzer]
    end
    
    subgraph "GUI Components"
        C[File Loading Section]
        D[Column Selection]
        E[Feature List]
        F[Feature Values]
        G[Visualization Area]
        H[Action Buttons]
    end
    
    subgraph "Data Processing"
        I[Statistical Features]
        J[Frequency Domain Features]
        K[Time Domain Features]
    end
    
    subgraph "Visualization"
        L[Time Series Plot]
        M[Feature Distribution]
        N[Correlation Matrix]
    end
    
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> G
    A --> H
    
    B --> I
    B --> J
    B --> K
    
    G --> L
    G --> M
    G --> N
```

## クラス構成

### LoadSensorAnalyzer
```mermaid
classDiagram
    class LoadSensorAnalyzer {
        -data: DataFrame
        -features: Dict
        -selected_features: List
        +load_data(file_path: str) bool
        +calculate_features(data_column: str, window_size: int) Dict
    }
```

**主要機能:**
- CSV/Excel ファイル読み込み
- **20+の特徴量抽出アルゴリズム**
  - 統計特徴量: mean, std, max, min, range, rms, skewness, kurtosis等
  - 周波数ドメイン特徴量: FFTベース解析
  - 時間ドメイン特徴量: rolling統計、変化点検出

### LoadSensorGUI
```mermaid
classDiagram
    class LoadSensorGUI {
        -analyzer: LoadSensorAnalyzer
        -current_file: str
        -plot_texture: texture
        +setup_gui()
        +select_file()
        +load_data()
        +calculate_features()
        +plot_time_series()
        +plot_feature_distribution()
        +plot_correlation_matrix()
        +save_features()
        +run()
    }
```

## GUI レイアウト構成

```mermaid
graph TD
    A[Main Window 1200x800] --> B[File Loading Section]
    A --> C[Column Selection Section]
    A --> D[Feature Display Area]
    A --> E[Visualization Buttons]
    
    B --> B1[File Path Input]
    B --> B2[Select File Button]
    B --> B3[Load Data Button]
    
    C --> C1[Data Column Combo]
    C --> C2[Window Size Input]
    C --> C3[Calculate Features Button]
    
    D --> D1[Feature List - Child Window 300x400]
    D --> D2[Feature Values - Child Window 300x400]
    D --> D3[Visualization Area - Child Window 500x400]
    
    D1 --> D1A[Checkboxes for Feature Selection]
    D2 --> D2A[Feature Values Display]
    D3 --> D3A[Matplotlib Plot Textures]
    
    E --> E1[Time Series Plot]
    E --> E2[Feature Distribution]
    E --> E3[Correlation Matrix]
    E --> E4[Save Features]
```

## データフロー

```mermaid
sequenceDiagram
    participant U as User
    participant G as LoadSensorGUI
    participant A as LoadSensorAnalyzer
    participant F as File System
    participant M as Matplotlib
    
    U->>G: Select File
    G->>F: File Dialog
    F-->>G: File Path
    
    U->>G: Load Data
    G->>A: load_data()
    A->>F: Read CSV/Excel
    F-->>A: DataFrame
    A-->>G: Success/Failure
    
    U->>G: Select Column & Calculate Features
    G->>A: calculate_features()
    A->>A: Statistical Analysis
    A->>A: FFT Processing
    A->>A: Rolling Statistics
    A-->>G: Features Dict
    
    G->>G: Update GUI (Checkboxes + Values)
    
    U->>G: Create Visualization
    G->>M: Create Plot
    M-->>G: Figure
    G->>G: Convert to DearPyGUI Texture
    G->>G: Display in Visualization Area
```

## 特徴量カテゴリ

### 統計特徴量 (10種類)
- mean, std, max, min, range
- rms, peak_to_peak, crest_factor
- skewness, kurtosis

### 周波数ドメイン特徴量 (2種類)
- spectral_centroid
- spectral_energy

### 時間ドメイン特徴量 (8種類)
- Rolling統計: mean_of_rolling_mean, std_of_rolling_mean, mean_of_rolling_std, std_of_rolling_std
- 変化点検出: mean_diff, std_diff, max_diff
- ゼロクロッシング: zero_crossing_rate

## 可視化機能

```mermaid
graph LR
    A[Matplotlib] --> B[FigureCanvasAgg]
    B --> C[RGBA Buffer]
    C --> D[DearPyGUI Texture]
    D --> E[GUI Display]
    
    F[Time Series Plot] --> A
    G[Feature Distribution] --> A
    H[Correlation Matrix] --> A
```

**対応プロット:**
- 時系列プロット
- 特徴量分布（最大4つの特徴量）
- 特徴量相関行列