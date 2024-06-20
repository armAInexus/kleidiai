//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>

template <class... Args>
void hello_benchmark(benchmark::State& state, Args&&... args) {
    volatile size_t a = 0;
    for (auto _ : state) {
        for (int i = 0; i < 100; i++) {
            a++;
        }
    }
}

BENCHMARK(hello_benchmark);

BENCHMARK_MAIN();
