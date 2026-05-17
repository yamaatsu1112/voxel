# algorithm

C++ / CUDA で書いている voxel / GPU アルゴリズム実験用のライブラリ。

中心は SVO / SVT 系データ構造の編集処理で、Hash DAG / SVDAG、CUDA 版の voxel 編集、scan / sort / hash table などの GPU primitive も含めている。

## Structure

```text
lib/include/algo/svt/   SVO / SVT / Hash DAG / SVDAG implementations
lib/include/algo/cuda/  GPU primitives such as scan, sort, and hash tables
test/                   Tests
bench/                  Benchmarks
docs/                   Notes for selected implementations
tools/                  Small analysis and conversion tools
```

## Requirements

- CMake 3.16+
- C++20 compatible compiler
- CUDA toolkit for CUDA tests and benchmarks

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Benchmark

### Selected results

以下は Google Benchmark の `manual_time`。

Voxel edit は、入力 voxel を leaf ごとの `LeafMask` にまとめてから、
未生成の SVO path を生成し、leaf の bit を更新して、一様になった subtree を
畳み込む。このうち path 生成のやり方を allocation policy として切り替えている。
表には、まとめた後の leaf 数を host に読み出し、その数だけ後段 kernel を起動する
`HostLeafCountDispatch` 構成の結果を載せている。

| Allocation policy | Sphere, 8.78M voxels | Random, 16.78M voxels |
|---|---:|---:|
| `PlainDepthwise` | 5.37 ms / 1.64G voxels/s | 68.90 ms / 243M voxels/s |
| `ScanDepthwise` | 5.15 ms / 1.70G voxels/s | 54.82 ms / 306M voxels/s |
| `AllDepth` | 5.08 ms / 1.73G voxels/s | 57.09 ms / 294M voxels/s |
| `CompactAllDepth` | 5.00 ms / 1.76G voxels/s | 43.55 ms / 385M voxels/s |

Policy の見方:

- `PlainDepthwise`: 各 depth で tree を見て欠けている child を集めて確保する基準実装。
- `ScanDepthwise`: 各 leaf で最初に欠けている depth を先に記録し、depth ごとに必要な request だけを prefix sum で詰める。
- `AllDepth`: 全 depth の生成 request を一度に集め、初期化、接続、counter 更新を一括で行う。
- `CompactAllDepth`: depth ごとの request record ではなく leaf ごとの request count を用いて path を生成し、中間データと kernel 起動回数を減らす。

Terminal edit は、SVO の構造に沿って一様に埋まる領域を 1 つの
terminal node として表し、球の境界だけを terminal leaf mask で表す。
8.78M voxels の球を約 26.76k terminal inputs で表現して、
`place_terminal_edits` 全体を測る。

| Edit representation | Case | Inputs | Covered voxels | Time | Effective throughput |
|---|---|---:|---:|---:|---:|
| Voxel edits | Sphere | 8.78M voxels | 8.78M | 5.34 ms | 1.64G voxels/s |
| Terminal edits | Aligned sphere | 26.76k terminals | 8.78M | 0.459 ms | 19.1G voxels/s |
| Terminal edits | Unaligned sphere | 26.76k terminals | 8.78M | 1.13 ms | 7.78G voxels/s |
| Terminal edits | Half-overlap sphere | 26.76k terminals | 8.78M | 0.566 ms | 15.5G voxels/s |

Terminal case の見方:

- `Aligned sphere`: terminal input を world offset なしで置く。
- `Unaligned sphere`: 同じ terminal input に `(1, 1, 1)` の world offset を付け、SVO cell 境界からずれた編集を測る。
- `Half-overlap sphere`: 先に中央の sphere を置いた SVO に、半径の半分だけ x 方向へずらした sphere を置き、既存 subtree との重なりを含む編集を測る。

### Environment

- GPU: NVIDIA GeForce RTX 3070, 8 GiB, compute capability 8.6, 220 W power limit
- NVIDIA driver: 595.71.05
- CUDA toolkit: 13.2.1 / nvcc 13.2.78
- CPU: Intel Core i7-10700, 8 cores / 16 threads, up to 4.8 GHz
- Memory: 15 GiB
- OS: Arch Linux, Linux 7.0.3-arch1-2 x86_64
- Compiler: GCC 16.1.1
- CMake: 4.3.2

### Run

```bash
cmake --build build
./build/bench/bench_cuda_svt
```

CUDA ベンチは Google Benchmark で書いている。ビルド時に CUDA compiler が見つかった場合だけ `bench/` 以下の CUDA benchmark target が作られる。

```bash
./build/bench/bench_cuda_svt_phases --benchmark_filter='.*(ScanDepthwise|CompactAllDepthHostLeafCountDispatch).*Random.*'
./build/bench/bench_cuda_svt --benchmark_filter='.*Place.*(ScanDepthwise|CompactAllDepthHostLeafCountDispatch).*'
./build/bench/bench_cuda_svt_terminal
./build/bench/bench_cuda_svt_terminal_phases
```

### Benchmarks

- `bench_cuda_svt`: SVO / SVT に対する voxel edit 全体のベンチ。dense / random な編集入力に対して、voxel edit から leaf mask への変換、place / destroy、get voxel query を測る。
- `bench_cuda_svt_phases`: SVO / SVT の voxel edit pipeline を段階ごとに測る。leaf mask build、leaf mask apply、uniform path collapse、allocation policy ごとの path allocation などを見るためのベンチ。
- `bench_cuda_svt_terminal`: voxel 単位の編集と terminal 単位の編集を比較するベンチ。球状の編集を、aligned / unaligned / half-overlap などの条件で測る。
- `bench_cuda_svt_terminal_phases`: terminal edit pipeline を段階ごとに測る。request 生成、allocation、write 適用、collapse / prune / release など、terminal edit の内訳を見るためのベンチ。
- `bench_cuda_hash_dag_gpu`: Hash DAG を GPU 上で編集する実験のベンチ。dense / random な place / destroy と get voxel query を測る。
- `bench_cuda_hash_table`: CUDA hash table 系のベンチ。slab hash set、bump slab hash set、acceleration hash set の insert と successful / unsuccessful contains を比較する。
- `bench_cuda_scan`: CUDA scan primitive のベンチ。inclusive / exclusive scan を複数実装と CUB で比較する。
- `bench_cuda_sort`: CUDA radix sort 系のベンチ。key-only sort、key-value pair sort、CUB との比較を行う。

### Filtering

各ベンチは Google Benchmark の `--benchmark_filter` で実行対象を絞れる。フィルタは benchmark name に対する正規表現として扱われる。

```bash
./build/bench/bench_cuda_svt --benchmark_filter=Place
./build/bench/bench_cuda_svt --benchmark_filter=Dense
./build/bench/bench_cuda_svt --benchmark_filter='.*Random.*1048576'
```

まず一覧だけ見たい場合は、Google Benchmark の `--benchmark_list_tests` を使う。

```bash
./build/bench/bench_cuda_svt --benchmark_list_tests
./build/bench/bench_cuda_svt_phases --benchmark_filter=AllocatePaths
./build/bench/bench_cuda_svt_terminal --benchmark_filter=Terminal
./build/bench/bench_cuda_hash_table --benchmark_filter=Contains
./build/bench/bench_cuda_scan --benchmark_filter=CUB
./build/bench/bench_cuda_sort --benchmark_filter='P32x4|CUB'
```
