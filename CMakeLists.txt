cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
project(tensorrt_jni LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
add_definitions(-DAPI_EXPORTS)

if(ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
endif()

set(BUILD_SHARED_LIBS ON)

find_package(CUDAToolkit REQUIRED)

find_package(JNI)
if(JNI_FOUND)
    # Fixes odd AWT dependency
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined"
)

# set(CUDA_DIR /usr/local/cuda)

# Include CUDA directories
# include_directories(${CUDA_DIR}/include)
# link_directories(${CUDA_DIR}/lib64)
# include_directories(src/main/native/include)


# Add source files
set(SOURCES
    src/main/native/cpp/tensorrt_jni.cpp
    src/main/native/cpp/YOLOv11.cpp
    src/main/native/cpp/preprocess.cu
)

# Add headers
set(HEADERS
    src/main/native/include/YOLOv11.h
    src/main/native/include/macros.h
    src/main/native/include/logging.h
    src/main/native/include/cuda_utils.h
    src/main/native/include/preprocess.h
    src/main/native/include/common.h
)



# Include TensorRT
# include_directories(${TENSORRT_DIR}/include)
# link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

set(CMAKE_INSTALL_RPATH "lib")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
# also valid: windowsx86-64

# type can be "", "debug", "static", or "staticdebug"
set(OPENCV_TYPE "")

# Download opencv, and save the path
include(FetchContent)
fetchcontent_declare(
    opencv_lib
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
)
fetchcontent_makeavailable(opencv_lib)

# download OpenCV headers
fetchcontent_declare(
    opencv_header
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
)
fetchcontent_makeavailable(opencv_header)

# This probably doesn't work great with shared libraries, but I don't care about those right now
file(
    GLOB_RECURSE OPENCV_LIB_PATH
    "${opencv_lib_SOURCE_DIR}/**/*.lib"
    "${opencv_lib_SOURCE_DIR}/**/*.so*"
)
set(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on opencv ${OPENCV_LIB_PATH}")

add_library(
    ${PROJECT_NAME}
    src/main/native/cpp/tensorrt_jni.cpp
    src/main/native/cpp/YOLOv11.cpp
    src/main/native/cuda/preprocess.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC 
    Threads::Threads 
    ${OPENCV_LIB_PATH} 
    CUDA::cudart
    ${TENSORRT_LIBS}
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        # ${CMAKE_CURRENT_SOURCE_DIR}
        # ${LIBRKNNRT_INCLUDES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/model_zoo
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/3rdparty/rga/RK3588/include
        ${JNI_INCLUDE_DIRS}
        # ${OPENCV_INCLUDE_PATH}
) 

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\\\$ORIGIN/")

# include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR}/lib)
# install(FILES ${TENSORRT_DIR} DESTINATION ${CMAKE_BINARY_DIR}/lib)

add_executable(main_test
    src/main/native/cpp/main_test.cpp 
    src/main/native/cpp/YOLOv11.cpp
    src/main/native/cuda/preprocess.cu
)

target_link_libraries(
    main_test
    PUBLIC 
    Threads::Threads 
    ${OPENCV_LIB_PATH} 
    CUDA::cudart
    ${TENSORRT_LIBS}
)

target_include_directories(
    main_test
    PRIVATE
        # ${CMAKE_CURRENT_SOURCE_DIR}
        # ${LIBRKNNRT_INCLUDES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/model_zoo
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include/3rdparty/rga/RK3588/include
        ${JNI_INCLUDE_DIRS}
        # ${OPENCV_INCLUDE_PATH}
)
