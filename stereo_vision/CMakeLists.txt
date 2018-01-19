cmake_minimum_required(VERSION 2.8)
project( VideoSegmentation )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( 1ref_test ref_test1.cpp)
target_link_libraries( 1ref_test ${OpenCV_LIBS} )

add_executable( 2ref_test ref_test2.cpp)
target_link_libraries( 2ref_test ${OpenCV_LIBS} )

add_executable( 3ref_test ref_test3.cpp)
target_link_libraries( 3ref_test ${OpenCV_LIBS} )

add_executable( 4ref_test ref_test4.cpp)
target_link_libraries( 4ref_test ${OpenCV_LIBS} )