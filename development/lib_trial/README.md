
# DID

- `uv add --editable . --dev`: dev 環境に編集可能なライブラリ (カレントディレクトリより) をインストール
- `uv sync --group dev`: dev グループになる
- `uv add pytest --dev`: pytest 追加

settings.json

```json
{
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.analysis.extraPaths": ["./src"],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

pyproject.toml

```toml
[project]
name = "lib-trial"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "ko", email = "ko@to.me" }
]
requires-python = ">=3.13"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "lib-trial",
    "pytest>=8.4.1",
]

[tool.uv.sources]
lib-trial = { workspace = true }

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```
