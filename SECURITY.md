<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Security Policy

KleidiAI software is verified for security for official releases and as such does not make promises about the quality of
the product for patches delivered between releases.

## Reporting a Vulnerability

Security vulnerabilities may be reported to the Arm Product Security Incident Response Team (PSIRT) by sending an email
to [psirt@arm.com](mailto:psirt@arm.com).

For more information visit https://developer.arm.com/support/arm-security-updates/report-security-vulnerabilities

## Security Guidelines

When KleidiAI is integrated and used in a product, developer must follow the security guidelines
to improve security of the product. The following guidelines are not comprehensive
and additional measures should be deployed to further protect the product.

- AI/ML model description and data must be sufficiently protected from unauthorized modification.
  KleidiAI micro-kernels perform AI/ML operation as defined in the API
  and does not provide a mechanism to detect and protect the system from malicious use of the API
  including (but not limited to) excessive use of CPU resources and denial-of-service attack.

- Optimizations in KleidiAI micro-kernels might introduce inaccuracy due to the use of floating-point
  arithmetic, fixed-point arithmetic and quantization techniques.
  If the decisions made by or based on the output of AI/ML model have security implication
  to the system, a safety envelop must be defined and deployed to make sure unexpected decisions
  do not lead to security issue.

- KleidiAI micro-kernels do not allocate memory but operate on the buffer allocated and shared
  by the caller of the API. The micro-kernels do not perform bound check, memory clean-up, etc.
  The caller must make sure that the data buffer has been sufficiently allocated,
  passed to the micro-kernels correctly and all its data handled properly.

- When KleidiAI is linked to the product as a shared library, the library and linking must be protected
  from unauthorized modification.

## Third Party Dependencies

Build scripts within this project download third party sources. KleidiAI uses the following third party sources:

- Google Test v1.14.0, for the testing suite.
- Google Benchmark v1.8.4, for the benchmarking suite.
