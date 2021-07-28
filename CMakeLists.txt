cmake_minimum_required(VERSION 3.18.0)

project(ML)

set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB ML_FILES_FOUND
    math/include/*.hpp
    math/src/*.cpp
	IO/include/*.hpp
    IO/src/*.cpp
	stats/include/*.hpp
	stats/src/*.cpp
	CNN/include/*.hpp
	CNN/src/*.cpp
    stats/layers/include/*.hpp
	third_party/openblas/include/*.h
	ML.cpp
    StdAfx.h)
	
add_executable(ML ${ML_FILES_FOUND})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/lodepng)

target_include_directories(ML
    PRIVATE
        .
    PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/math/include
		${CMAKE_CURRENT_SOURCE_DIR}/IO/include
		${CMAKE_CURRENT_SOURCE_DIR}/stats/include
		${CMAKE_CURRENT_SOURCE_DIR}/CNN/include
		${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas/include
		${CMAKE_CURRENT_SOURCE_DIR}/third_party/lodepng/include
		${CMAKE_CURRENT_SOURCE_DIR}/stats/layers/include)
		
target_link_libraries(ML
    PUBLIC
		libopenblas
        lodepng)
add_library(libopenblas third_party/openblas/libopenblas.dll)
set_target_properties(libopenblas PROPERTIES LINKER_LANGUAGE C)
		
foreach(_source ${ML_FILES_FOUND})
	if (IS_ABSOLUTE "${_source}")
		file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
	else()
		set(_source_rel "${_source}")
	endif()
	get_filename_component(_source_path "${_source_rel}" PATH)
	string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
	source_group("${_source_path_msvc}" FILES "${_source}")
endforeach()

if(MSVC)
    target_compile_options(ML PRIVATE /bigobj /WX /W0)
endif(MSVC)