<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm® Limited and/or its affiliates <open-source-office@Arm®.com>

    SPDX-License-Identifier: Apache-2.0
-->

<h1><b>KleidiAI</b></h1>

KleidiAI is an open-source library that provides optimized performance-critical routines, also known as <strong>micro-kernels</strong>, for artificial intelligence (AI) workloads tailored for Arm® CPUs.

These routines are tuned to exploit the capabilities of specific Arm® hardware architectures, aiming to maximize performance.

The KleidiAI library has been designed for ease of adoption into C or C++ machine learning (ML) and AI frameworks. Specifically, developers looking to incorporate specific micro-kernels into their projects can only include the corresponding <strong>.c</strong> and <strong>.h</strong> files associated with those micro-kernels and a common header file.

> ⚠️ Since the project is currently in an experimental phase and actively evolving, it is essential to note that API modifications, including function name changes, and feature enhancements may occur without advance notice. The development team is committed to refining the library to meet production standards.

<h1> Who is this library for? </h1>

KleidiAI is a library for AI/ML framework developers interested in accelerating the computation on Arm® CPUs.

<h1> What is a micro-kernel? </h1>

A micro-kernel, or <strong>ukernel</strong>, can be defined as a near-minimum amount of software to accelerate a given ML operator with high performance.

For example, consider the convolution 2d operator performed through the Winograd algorithm. In this case, the computation requires the following four operations:

- Winograd input transform
- Winograd filter transform
- Matrix multiplication
- Winograd output transform

Each of the preceding operations is a micro-kernel. For an example, please refer to the [first micro kernel PR](https://gitlab.arm.com/kleidi/kleidiai/-/merge_requests/2)

<em>However, why are the preceding operations not called kernels or functions?</em>

<b>Because the micro-kernels are designed to give the flexibility to process also a portion of the output tensor</b>, which is the reason why we call it micro-kernel.

> ℹ️ The API of the micro-kernel is intended to provide the flexibility to dispatch the operation among different working threads or process only a section of the output tensor. Therefore, the caller can control what to process and how.

A micro-kernel exists for different Arm® architectures, technologies, and computational parameters (for example, different output tile sizes). These implementations are called <strong>micro-kernel variants</strong>. All micro-kernel variants of the same micro-kernel type perform the same operation and return the same output result.

<h1> Key features </h1>

Some of the key features of KleidiAI are the following:

- No dependencies on external libraries
- No internal memory allocation​
- No internal threading mechanisms
- Stateless, stable, and consistent API​
- Performance-critical compute-bound and memory-bound micro-kernels
- Specialized micro-kernels for different Arm® CPU architectures and technologies
- Specialized micro-kernels for different fusion patterns
- Micro-kernel as a standalone library, consisting of only a <strong>.c</strong> and <strong>.h</strong> files

> ℹ️ The micro-kernel API is designed to be as generic as possible for integration into third-party runtimes.

<h1> Frequently Asked Questions (FAQ) </h1>

<h2> What is the difference between the Compute Library for the Arm® Architecture (ACL) and KleidiAI? </h2>

This question will pop up naturally if you are familiar with the **[ACL](https://github.com/ARM-software/ComputeLibrary)**.

<em>ACL and KleidiAI differ with respect to the integration point into the AI/ML framework</em>.

ACL provides a complete suite of ML operators for Arm® CPUs and Arm Mali™ GPUs. It also provides a runtime with memory management, thread management, fusion capabilities, etc.

Therefore, <strong>ACL is a library suitable for frameworks that need to delegate the model inference computation entirely</strong>.

KleidiAI offers performance-critical operators for ML, like matrix multiplication, pooling, depthwise convolution, and so on. As such, <strong>KleidiAI is designed for frameworks where the runtime, memory manager, thread management, and fusion mechanisms are already available</strong>.

<h2> Can the micro-kernels be multi-threaded? </h2>

<strong>Yes, they can</strong>. The micro-kernel can be dispatched among different threads using the thread management available in the target AI/ML framework.

<em>The micro-kernel does not use any internal threading mechanism</em>. However, the micro-kernel's API is designed to allow the computation to be carried out only on specific areas of the output tensor. Therefore, this mechanism is sufficient to split the workload on parallel threads. More information on dispatching the micro-kernels among different threads will be available soon.

<h1> How to contribute </h1>

KleidiAI is not currently accepting contributions during the bring up phase.

<h1> License </h1>

[Apache-2.0](LICENSES/Apache-2.0.txt).
