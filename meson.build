project(
    'cndetector',
    'cpp',
    version: run_command(
        find_program('python3', 'python', required: true),
        meson.current_source_dir() / 'buildtools' / 'get_version.py',
        check: true,
    ).stdout().strip(),
    default_options: [
        'cpp_std=c++20',
        'optimization=3',
        'b_ndebug=if-release',
        'warning_level=3',
    ]
)

compiler = meson.get_compiler('cpp')

if host_machine.system() == 'windows' and compiler.get_id() == 'gcc'
    # MinGW only: statically link the GCC / libstdc++ / winpthread runtimes so
    # the resulting .pyd is self-contained and loads into the (MSVC-built)
    # official CPython without needing libgcc_s_*.dll, libstdc++-6.dll or
    # libwinpthread-1.dll on PATH. (MSVC handles the CRT via b_vscrt=mt and
    # needs none of these flags.)
    add_project_link_arguments(
        '-static', '-static-libgcc', '-static-libstdc++', language: 'cpp')
endif

if host_machine.system() != 'windows'
    add_project_arguments('-flto', language: 'cpp')
    add_project_link_arguments('-flto', language: 'cpp')
endif

if host_machine.system() != 'windows'
    add_project_arguments('-Werror', language: 'cpp')
endif

SRC_DIR = 'cndetector' / 'cpp'
INCLUDES = include_directories(SRC_DIR)

# ParallelFor.h spawns std::thread workers, so every target that pulls it in
# needs the platform thread library (-pthread on Linux).
threads_dep = dependency('threads')

libgraph = static_library(
    'graph',
    [
        SRC_DIR / 'Graph' / 'CNP_Graph.cpp',
        SRC_DIR / 'Graph' / 'DCNP_Graph.cpp',
    ],
    include_directories: INCLUDES,
    dependencies: threads_dep,
)

libsearch = static_library(
    'search',
    [
        SRC_DIR / 'search' / 'L2NSSearch.cpp',
        SRC_DIR / 'search' / 'DCNPSearch.cpp',
    ],
    include_directories: INCLUDES,
    link_with: libgraph,
)

libcrossover = static_library(
    'crossover',
    [
        SRC_DIR / 'crossover' / 'reduceSolveCombine.cpp',
    ],
    include_directories: INCLUDES,
    link_with: [libgraph, libsearch],
)

libcore = static_library(
    'core',
    [
        SRC_DIR / 'Population.cpp',
        SRC_DIR / 'ProblemData.cpp',
    ],
    include_directories: INCLUDES,
    dependencies: threads_dep,
    link_with: [libgraph, libsearch, libcrossover],
)

py = import('python').find_installation()
dependencies = [py.dependency(), dependency('pybind11'), threads_dep]

py.extension_module(
    '_cndetector',
    [SRC_DIR / 'pybind.cpp'],
    dependencies: dependencies,
    link_with: [libcore, libgraph, libsearch, libcrossover],
    install: true,
    subdir: 'cndetector',
    include_directories: INCLUDES,
)
