cmake_minimum_required(VERSION 3.18)
project(AlgoPainter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)
find_package(CUDAToolkit QUIET)
find_package(OpenMP QUIET)

set(CORE_SOURCES
		src/main.cpp
		src/Application.cpp
		src/Rasterizer.cpp
		src/GAUtils.cpp
		src/Gene.cpp
		src/Fitness.cpp
		src/FontAnalyzer.cpp
		src/IRasterizer.cpp
)

set(HEADERS
		src/Application.h
		src/Pixel.h
		src/Rasterizer.h
		src/Individual.h
		src/Gene.h
		src/GAUtils.h
		src/FontAnalyzer.h
		src/Fitness.h
		src/CudaRasterizer.h
		src/IRasterizer.h
)

if (CUDAToolkit_FOUND)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)

	list(APPEND CORE_SOURCES
			src/CudaRasterizer.cpp
			src/CudaRasterizerKernels.cu
	)
endif ()

add_executable(${PROJECT_NAME} ${CORE_SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PRIVATE
		SFML::Graphics
		SFML::Window
		SFML::System
)

if (CUDAToolkit_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
endif ()

if (OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENMP)
endif ()

if (WIN32)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			$<TARGET_FILE:SFML::Graphics>
			$<TARGET_FILE:SFML::Window>
			$<TARGET_FILE:SFML::System>
			$<TARGET_FILE_DIR:${PROJECT_NAME}>
			COMMENT "Copying SFML DLLs to output directory..."
	)
endif ()

# ============================================
# Code Formatting Target (clang-format)
# ============================================
find_program(CLANG_FORMAT_EXE NAMES clang-format)

if (CLANG_FORMAT_EXE)
	file(GLOB_RECURSE ALL_SOURCE_FILES
			${CMAKE_SOURCE_DIR}/src/*.cpp
			${CMAKE_SOURCE_DIR}/src/*.h
			${CMAKE_SOURCE_DIR}/src/*.cu
	)

	add_custom_target(format
			COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
			COMMENT "Running clang-format on all source files..."
			VERBATIM
	)

	add_custom_target(format-check
			COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${ALL_SOURCE_FILES}
			COMMENT "Checking code formatting..."
			VERBATIM
	)
else ()
	message(STATUS "clang-format not found - format targets will not be available")
endif ()

