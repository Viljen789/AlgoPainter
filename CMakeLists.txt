cmake_minimum_required(VERSION 3.18)
project(AlgoPainter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Point to your SFML 3 folder
set(SFML_DIR "C:/Libs/SFML-3.0.2/lib/cmake/SFML")

find_package(SFML 3.0 COMPONENTS Graphics Window System REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/main.cpp
    src/Application.cpp
    src/CudaRasterizer.cpp
    src/Fitness.cpp
    src/GAUtils.cpp
    src/Gene.cpp
    src/Rasterizer.cpp
)

set(HEADERS
    src/Application.h
    src/CudaRasterizer.h
    src/Fitness.h
    src/GAUtils.h
    src/Gene.h
    src/Individual.h
    src/Pixel.h
    src/Rasterizer.h
)

set(CUDA_SOURCES
    src/CudaRasterizerKernels.cu
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
    ${CUDA_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}
    SFML::Graphics
    SFML::Window
    SFML::System
    CUDA::cudart
)

# --- NEW SECTION: Copy SFML DLLs to the executable folder automatically ---
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SFML::Graphics>
    $<TARGET_FILE:SFML::Window>
    $<TARGET_FILE:SFML::System>
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying SFML DLLs to output directory..."
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)
