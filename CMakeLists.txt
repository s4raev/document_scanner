cmake_minimum_required(VERSION 3.15)
project(document_scanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
add_subdirectory(${IMGUI_DIR})

set(NFD_DIR ${CMAKE_SOURCE_DIR}/external/nativefiledialog)

add_library(nfd STATIC
    ${NFD_DIR}/src/nfd_common.c
    ${NFD_DIR}/src/nfd_cocoa.m
)

target_include_directories(nfd PUBLIC
    ${NFD_DIR}/src/include
    ${NFD_DIR}/src       
)

if(APPLE)
    find_library(COCOA_LIBRARY Cocoa)
    target_link_libraries(nfd PRIVATE ${COCOA_LIBRARY})
    
    set_source_files_properties(${NFD_DIR}/src/nfd_cocoa.m PROPERTIES
        COMPILE_FLAGS "-x objective-c"
    )
endif()

find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(SOURCES
    src/main.cpp
    src/document_scanner.cpp
    src/image_processing.cpp
)

add_executable(document_scanner ${SOURCES})

target_include_directories(document_scanner PRIVATE
    include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${NFD_DIR}/src/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(document_scanner PRIVATE
    imgui
    nfd
    glfw
    ${OpenCV_LIBS}
    OpenGL::GL
)
