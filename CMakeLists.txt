cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME linear_histogram_stretching)
project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

set(COPY_RESOURCE copy_resource)
add_custom_target(${COPY_RESOURCE} ALL)
add_custom_command(
        TARGET ${COPY_RESOURCE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/resource/
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/resource/
	COMMAND ${CMAKE_COMMAND} -E copy_directory
          	${CMAKE_SOURCE_DIR}/resource/
          	${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/resource
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})

add_executable(${PROJECT_NAME}_sample main_sample.cpp)
target_link_libraries(${PROJECT_NAME}_sample ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_sample ${MPI_C_LIBRARIES})

add_executable(${PROJECT_NAME}_test main_test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_test ${MPI_C_LIBRARIES})