cmake_minimum_required(VERSION 2.9)

set(TARGET_BUILD_PLATFORM x64)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Add OpenCV:
set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)
set(OpenCV_FOUND ON)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

project(RobustMatting)

include_directories(./libs/eigen-3.3.3/)

#add_subdirectory(./libs/utils/)
#add_subdirectory(./src/generalar/)

file(GLOB ROBUSTMATTING_SRC
	./src/robust_matting/*.h
	./src/robust_matting/*.cpp
)
add_executable(RobustMatting ${ROBUSTMATTING_SRC})

if(MSVC AND OpenCV_STATIC)
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

target_link_libraries(RobustMatting  
	${OpenCV_LIBS}
)

set(BINARY_DEBUG_DIR ${PROJECT_BINARY_DIR}/Debug/)
set(BINARY_RELEASE_DIR ${PROJECT_BINARY_DIR}/Release/)
file(MAKE_DIRECTORY ${BINARY_DEBUG_DIR})
file(MAKE_DIRECTORY ${BINARY_RELEASE_DIR})

file (GLOB DATA_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/data/*.*
)
file (COPY ${DATA_FILES} DESTINATION ${PROJECT_BINARY_DIR}/data/)