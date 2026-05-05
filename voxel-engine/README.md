# voxel-engine

Rust / Vulkan で書いている voxel engine。

`algorithm/` にある CUDA / C++ の研究実装そのものではなく、voxel の見た目や編集結果を確認するための実験環境として置いている。現在は SVO 系の voxel 表現、raycast、簡単な編集・描画確認などを扱う。

将来的には、CUDA 側で検証している SVO 編集処理を Vulkan compute / Rust 側へ移植し、インタラクティブに大規模な voxel の変化を確認できる環境にする。

## Structure

```text
engine/  Vulkan renderer, ECS, input, UI, voxel modules
game/    engine を使った実行用アプリケーション
```

## Voxel-related paths

voxel 関連の Rust 側の中心は `engine/voxel/src/voxel/` にある。

```text
engine/voxel/src/voxel/      voxel object, command, manager, raycast data
engine/voxel/src/voxel/svo/  SVO backend and variants
```

SVO variant は `engine/voxel/src/voxel/svo/` 以下に分かれている。

```text
engine/voxel/src/voxel/svo/individual/       individual node layout
engine/voxel/src/voxel/svo/grouped/          grouped node layout
engine/voxel/src/voxel/svo/sv64_individual/  64-way branching individual layout
engine/voxel/src/voxel/svo/sv64_grouped/     64-way branching grouped layout
```

voxel shader は共通shaderとvariantごとのshaderに分かれている。

```text
engine/voxel/src/shaders/svo/                  shared SVO shader code
engine/voxel/src/voxel/svo/individual/shaders/ individual variant shaders
engine/voxel/src/voxel/svo/grouped/shaders/    grouped variant shaders
engine/voxel/src/voxel/svo/sv64_individual/shaders/
engine/voxel/src/voxel/svo/sv64_grouped/shaders/
```

## Requirements

- Linux
- Rust toolchain
- Vulkan 対応GPUとdriver
- Slang shader compiler を使える環境

## Run

リポジトリルートではなく、`game/` から release ビルドで起動する。

```bash
cd voxel-engine/game
cargo run --release
```

## 起動後の操作

現在はまだ UI を整備していないため、メニューは画像を貼り付けただけの状態になっている。

- 起動後に表示される長方形のうち、一番上を左クリックすると地形生成のシーンへ移行する。
- 地形生成のシーンでESCを押すと2つの長方形が表示される。上側のボタンを左クリックすると元のシーンにもどる。下側のボタンを押すとメニューに戻ることを想定しているが、現在はバグがあり正常に動作しない。
- アプリの終了はOS側から行う。