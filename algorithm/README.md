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

```bash
cmake --build build
./build/bench/bench_cuda_svt
```

CUDA ベンチは Google Benchmark で書いている。ビルド時に CUDA compiler が見つかった場合だけ `bench/` 以下の CUDA benchmark target が作られる。

### Recommended

```bash
./build/bench/bench_cuda_svt_phases --benchmark_filter='.*(ScanDepthwise|CompactAllDepthHostLeafCountDispatch).*Random.*'
./build/bench/bench_cuda_svt --benchmark_filter='.*Place.*(ScanDepthwise|CompactAllDepthHostLeafCountDispatch).*'
./build/bench/bench_cuda_svt_terminal
./build/bench/bench_cuda_svt_terminal_phases
```

`bench_cuda_svt_phases` は、allocation policy として `ScanDepthwise` / `CompactAllDepthHostLeafCountDispatch` を比較し、入力は `Random` に絞る。`bench_cuda_svt` は同じ policy の `Place` 系ベンチを見る。terminal edit 系はまず全体を見るため、filter なしで実行する。

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
