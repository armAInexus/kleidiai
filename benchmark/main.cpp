//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "benchmark/matmul/matmul_f32.hpp"

void print_usage(char* name) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "%s -m 13 -n 17 -k 18\n", name);
    fprintf(stderr, "\n");
    fprintf(stderr, "For additional options:\n");
    fprintf(stderr, "%s --help\n", name);
}

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);

    bool mflag = false;
    bool nflag = false;
    bool kflag = false;

    size_t m = 0;
    size_t n = 0;
    size_t k = 0;

    int opt;
    while ((opt = getopt(argc, argv, "m:n:k:")) != -1) {
        switch (opt) {
            case 'm':
                m = atoi(optarg);
                mflag = true;
                break;
            case 'n':
                n = atoi(optarg);
                nflag = true;
                break;
            case 'k':
                k = atoi(optarg);
                kflag = true;
                break;
            case '?':
                // Fallthrough
            default:
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!mflag || !nflag || !kflag) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    kai_matmul matmul_f32;
    for (int i = 0; i < num_ukernel_variants; i++) {
        ::benchmark::RegisterBenchmark(ukernel_variants[i].name, matmul_f32, ukernel_variants[i], m, n, k);
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
