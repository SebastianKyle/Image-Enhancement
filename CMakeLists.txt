cmake_minimum_required(VERSION 3.6.0)
project(ImageEnhancement VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(
    $(CMAKE_SOURCE_DIR)/header
    $(CMAKE_SOURCE_DIR)/src
)

set(SOURCES
    main.cpp
    src/lib.cpp
    src/PointWiseTransformer.cpp
    src/HistogramTransformer.cpp
    src/NoiseSmoother.cpp
)

set(CMAKE_CXX_STANDARD 11)

add_executable(ImageEnhancement ${SOURCES})

target_link_libraries( ImageEnhancement ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
