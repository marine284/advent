Optimized for AVX2

cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

taskset -c 2 build/_2025

llvm-mca -mcpu=znver3 -resource-pressure -instruction-info -bottleneck-analysis -dispatch-stats -iterations=100 -timeline build/generated/main.s
