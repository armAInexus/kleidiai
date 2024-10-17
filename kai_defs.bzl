#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Build definitions for KleidiAI"""

# Extra warnings for GCC/CLANG C/C++
def kai_gcc_warn_copts():
    return [
        "-Wall",
        "-Wdisabled-optimization",
        "-Werror",
        "-Wextra",
        "-Wformat-security",
        "-Wformat=2",
        "-Winit-self",
        "-Wno-ignored-attributes",
        "-Wno-misleading-indentation",
        "-Wno-overlength-strings",
        "-Wstrict-overflow=2",
        "-Wswitch-default",
        "-Wno-vla",
    ]

def kai_gcc_warn_cxxopts():
    return kai_gcc_warn_copts() + [
        "-Wctor-dtor-privacy",
        "-Weffc++",
        "-Woverloaded-virtual",
        "-Wsign-promo",
    ]

# GCC/CLANG compiler options
def kai_gcc_std_copts():
    return ["-std=c99"] + kai_gcc_warn_copts()

# GCC/CLANG compiler options
def kai_gcc_std_cxxopts():
    return ["-std=c++17"] + kai_gcc_warn_cxxopts()

def kai_cpu_select(cpu_uarch):
    if len(cpu_uarch) == 0:
        return "armv8-a"
    else:
        return "armv8.2-a" + cpu_uarch

def kai_cpu_i8mm():
    return "+i8mm"

def kai_cpu_dotprod():
    return "+dotprod"

def kai_cpu_bf16():
    return "+bf16"

def kai_cpu_fp16():
    return "+fp16"

def kai_cpu_neon():
    return ""

def kai_cpu_sme():
    return "+sve+sve2"

def kai_cpu_sme2():
    return "+sve+sve2"

# MSVC compiler options
def kai_msvc_std_copts():
    return ["/Wall"]

def kai_msvc_std_cxxopts():
    return ["/Wall"]

def kai_copts(ua_variant):
    return select({
        "//:windows": kai_msvc_std_copts(),
        # Assume default to use GCC/CLANG compilers. This is a fallback case to make it
        # easier for KleidiAI library users
        "//conditions:default": kai_gcc_std_copts() + ["-march=" + kai_cpu_select(ua_variant)],
    })

def kai_cxxopts(ua_variant):
    return select({
        "//:windows": kai_msvc_std_cxxopts(),
        # Assume default to use GCC/CLANG compilers. This is a fallback case to make it
        # easier for KleidiAI library users
        "//conditions:default": kai_gcc_std_cxxopts() + ["-march=" + kai_cpu_select(ua_variant)],
    })

def kai_c_library(name, **kwargs):
    """C native cc_library wrapper with custom parameters and defaults

    Args:
        name (string): name of target library
        **kwargs (dict): other arguments like srcs, hdrs, deps
    """
    kwargs["copts"] = kwargs.get("copts", []) + kai_copts(kwargs.get("cpu_uarch", kai_cpu_neon()))
    kwargs["deps"] = ["//:common"] + kwargs.get("deps", [])
    kwargs["linkstatic"] = kwargs.get("linkstatic", True)

    # Remove custom cpu_uarch paramter before passing it to cc_library
    if "cpu_uarch" in kwargs:
        kwargs.pop("cpu_uarch")

    native.cc_library(
        name = name,
        **kwargs
    )

def kai_cxx_library(name, **kwargs):
    """C++ native cc_library wrapper with custom parameters and defaults

    Args:
        name (string): name of target library
        **kwargs (dict): other arguments like srcs, hdrs, deps
    """
    kwargs["copts"] = kwargs.get("copts", []) + kai_cxxopts(kwargs.get("cpu_uarch", kai_cpu_neon()))
    kwargs["deps"] = ["//:common"] + kwargs.get("deps", [])
    kwargs["linkstatic"] = kwargs.get("linkstatic", True)

    # Remove custom cpu_uarch paramter before passing it to cc_library
    if "cpu_uarch" in kwargs:
        kwargs.pop("cpu_uarch")

    native.cc_library(
        name = name,
        **kwargs
    )
