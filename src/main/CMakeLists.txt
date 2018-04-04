message(STATUS "============main module===========")
# include头文件目录
include_directories( ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                   )

# library目录及name名称
link_directories(${PROJECT_SOURCE_DIR}/lib ${PROJECT_SOURCE_DIR}/3rdparty/lib)


# test picture execute file
add_executable(test_picture test_picture.cpp)
target_link_libraries(test_picture mtcnn ${OpenCV_LIBS})
set_target_properties(test_picture
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

# test camera execute file
add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera mtcnn ${OpenCV_LIBS})
set_target_properties(test_camera
        PROPERTIES 
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)


