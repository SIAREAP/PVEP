cmake_minimum_required(VERSION 3.8)
project(competition_tutorials)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ariac_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION lib/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  launch/start_competition.py
  launch/start_competition.launch.py
  competition_tutorials/sia_interface.py
  competition_tutorials/ariac_interface.py
  competition_tutorials/test.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY 
  config
  launch
  resources
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
