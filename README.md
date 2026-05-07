# Voxel Research Notes

面談用

個人的に進めている voxel / GPU アルゴリズム実験のリポジトリ。

主な関心は、voxel データを大規模に生成・変化させるためのデータ構造と、その編集処理を GPU 上で高速に実行する方法にある。現在は特に Sparse Voxel Octree (SVO) 周辺の編集アルゴリズム、CUDA 実装、Hash DAG への応用を試している。

## 構成

### `algorithm/`

C++ / CUDA で書いているアルゴリズム実験の中心。

SVO / SVT、Hash DAG、SVDAG、GPU 用の scan / sort / hash table などを実装・検証している。高度な実装やベンチマークは主にこちらにある。

主な内容:

- `lib/include/algo/svt/`: SVO / SVT / Hash DAG / SVDAG 系の実装
- `lib/include/algo/svt/cuda/`: CUDA 版の voxel / terminal 編集処理
- `lib/include/algo/svt/hash_dag_gpu/`: Hash DAG を GPU 上で扱う実験
- `lib/include/algo/cuda/`: scan, sort, hash table などの GPU primitive
- `test/`: データ構造と CUDA 実装のテスト
- `bench/`: CUDA 実装やデータ構造のベンチマーク

ビルド方法やベンチマークの内容は [algorithm/README.md](algorithm/README.md) に置いている。

### `voxel-engine/`

Rust / Vulkan で書いている voxel engine。

voxel の見た目や編集結果を確認するための実験環境。現在は SVO 系の voxel 表現、raycast、簡単な編集・描画確認などを扱っている。

CUDA 側で検証している編集アルゴリズムを、将来的に Vulkan compute / Rust 側へ移して、大規模な voxel の変化を確認できる環境にすることを目指している。

起動方法や構成は [voxel-engine/README.md](voxel-engine/README.md) に置いている。

## 今後やりたいこと

- CUDAで実装しているSVO編集処理をVulkan computeへ移植する
- 地形生成に限らず、コンピュータ側で大規模に voxel を変化させる処理を扱う
- Hash DAG / SVDAGの共有構造に対して、GPU上で編集を適用する方法を試す
- `algorithm/` で検証した処理を `voxel-engine/` で可視化し、動作確認できるようにする
- マテリアルの扱いを考える

- voxelが本当に有用か検討
- voxelデータからメッシュを生成して描画
- pixel単位ではなくvoxel単位で色を決定する
