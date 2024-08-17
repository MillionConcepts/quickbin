cmake_minimum_required(VERSION 3.26)
project(quickbin C)
set(CMAKE_C_STANDARD 11)

# TODO: it would be better for this to be portable to non-conda environments!
set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
set(Python_BINARY "${Python_ROOT_DIR}/bin/python")
find_package(Python COMPONENTS Interpreter Development NumPy)
add_executable(quickbin quickbin/src/quickbin_core.c quickbin/src/quickbin_core.h)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})
# 'm' is libm for math.h, mathcalls.h, etc.
target_link_libraries(quickbin PUBLIC ${Python_LIBRARIES} Python::NumPy m)
add_custom_command(
    TARGET quickbin
    POST_BUILD
    COMMAND ${Python_BINARY} setup.py build_ext --inplace && ${Python_ROOT_DIR}/bin/pip install -e .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
