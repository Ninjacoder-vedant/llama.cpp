ggml_add_backend_library(ggml-xlns
    ggml-xlns.cpp
)

# Tell CMake where to look for the xlnscpp files (the current directory)
target_include_directories(ggml-xlns PRIVATE .)