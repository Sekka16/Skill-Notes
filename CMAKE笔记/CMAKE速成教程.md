# CMAKE速成教程

## 01-basic

### A-hello-cmake

#### 1 CMakeList编写

- 设定cmake的最小版本：`cmake_minimum_required()`
- 设定项目的名字：`project()`
- 添加一个可执行文件：`add_executable()`

```cmake
# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.0)

# Set the project name
project (hello_cmake)

# Add an executable
add_executable(hello_cmake main.cpp)
```

可执行文件名通常与项目名称相同，**因此通常用`${PROJECT_NAME}`**

#### 2 In-place build与Out-of-soure build

为了保持源代码以及cmake产生的文件分离，因此通常不在原根目录下cmake，而是新建一个build文件夹，将cmake产生的文件放入build文件夹，即Out-of-soure build

具体来说：

```terminal
mkdir build

cd build

cmake ..
```

### B-hello-headers

#### 1 目录路径

| 变量                     | 含义                                              |
| ------------------------ | ------------------------------------------------- |
| CMAKE_SOURCE_DIR         | 源代码根目录路径                                  |
| CMAKE_CURRENT_SOURCE_DIR | 源代码当前目录路径                                |
| PROJECT_SOURCE_DIR       | 项目根目录路径（通常=CMAKE_SOURCE_DIR）           |
| CMAKE_BINARY_DIR         | 生成的二进制文件目录                              |
| CMAKE_CURRENT_BINARY_DIR | 生成的当前二进制文件目录                          |
| PROJECT_BINARY_DIR       | 项目生成的二进制文件目录（通常=CMAKE_BINARY_DIR） |

#### 2 源文件变量

设置一个变量存放源文件的路径，例如：

```cmake
set(SOURCES
    src/Hello.cpp
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
```

> 一个可选择的策略是使用**“file(GLOB ...)”**去匹配源文件，例如：file(GLOB SOURCES "src/*.cpp")

#### 3 包含目录

- 使用target_include_directories()命令包含目录

```cmake
# target为目标可执行文件名
target_include_directories(target
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)
```

#### 4 Make详细输出

```terminal
make VERBOSE=1
```

### C-static-library

#### 1 添加一个静态库

- add_library(xxx STATIC xxx)

```cmake
set(library_SOURCES
    src/Hello.cpp
)

add_library(hello_library STATIC ${library_SOURCES})
```

#### 2 设置目标文件的包含目录

- target_include_directories()

```cmake
# hello_library即为目标文件
target_include_directories(hello_library
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
```

- 作用域：

  - PRIVATE：该目录仅被添加到该目标中
  - PUBLIC：该目录不仅被添加到该目标中，还被添加到任何链接到该目录的目标中
  - INTERFACE：该目录被添加到任何链接到该目录的目标中

  > 关于INTERFACE的解释：例如有一个头文件math_utils.h，该文件定义了各个函数的接口但是并没有实现，即没有对应的cpp实现，其他的目标在链接该目录时需要做出自己的实现，INTERFACE的本质就是定义一个公共接口

#### 3 链接一个静态库

- target_link_libraries()

```cmake
add_executable(hello_binary ${binary_SOURCES})

target_link_libraries(hello_binary
   PRIVATE  hello_library
)
```

### D-shared-library

#### 1 添加一个动态库

- add_library(xxx SHARED xxx)

```cmake
set(library_SOURCES
    src/Hello.cpp
)

add_library(hello_library SHARED ${library_SOURCES})
```

#### 2 别名目标

- add_library(xxx ALIAS xxx)

```cmake
add_library(hello::library ALIAS hello_library)
```

#### 3 链接一个动态库

```cmake
add_executable(hello_binary ${binary_SOURCES})

target_link_libraries( hello_binary
    hello::library
)
```

### E-installing

#### 1 安装命令

- install()命令：默认路径存放在`CMAKE_INSTALL_PREFIX`变量中，可通过`cmake .. -DCMAKE_INSTALL_PREFIX=/install/location`更改

```cmake
# 将从目标cmake_examples_inst_bin目标生成的二进制文件安装到目标${cmake_Install_PREFIX}/bin
install (TARGETS cmake_examples_inst_bin
    DESTINATION bin)
# 将从目标cmake_examples_inst目标生成的共享库安装到目标${cmake_Install_PREFIX}/lib
install (TARGETS cmake_examples_inst
    LIBRARY DESTINATION lib)
# 将配置文件安装到目标$｛CMAKE_Install_PRIIX｝/etc
install (FILES cmake-examples.conf
    DESTINATION etc)
```

> 共享库命令在windows上无效，在windows上的dll文件需要用命令：
>
> install (TARGETS cmake_examples_inst
>     LIBRARY DESTINATION lib
>     RUNTIME DESTINATION bin)

- 运行make install后，CMake会生成一个install_manifest.txt文件，其中包含所有已安装文件的详细信息

> 如果以root身份运行make-install命令，install_manifest.txt文件将归root所有。

#### 2 运行安装的二进制文件

```terminal
# 这里的 $LD_LIBRARY_PATH 是一个环境变量，它代表当前已经设置的共享库搜索路径。
# ":/usr/local/lib"是要添加到 LD_LIBRARY_PATH 中的新路径，即 /usr/local/lib。
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib cmake_examples_inst_bin
```

#### 3 改变安装的默认位置

- 在顶层CMakeLists下添加：

```cmake
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  message(STATUS "Setting default CMAKE_INSTALL_PREFIX path to ${CMAKE_BINARY_DIR}/install")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING "The path to use for make install" FORCE)
endif()
```

- DESDIR参数

```terminal
make install DESTDIR=/tmp/stage
```

这将为所有的安装文件创建路径 `${DESTDIR}/${CMAKE_INSTALL_PREFIX}`。在这个例子中，它会将所有文件安装在路径 `/tmp/stage/usr/local` 下。

#### 4 卸载

```terminal
sudo xargs rm < install_manifest.txt
```

### F-build-type

#### 1 优化级别

CMake内置了一些构建配置，指定优化级别以及是否在二进制文件中包含调试信息，级别有：

- **Release（发布）** - 为编译器添加 -O3 -DNDEBUG 标志
- **Debug（调试）** - 添加 -g 标志
- **MinSizeRel（最小尺寸发布）** - 添加 -Os -DNDEBUG 标志
- **RelWithDebInfo（带调试信息的发布）** - 添加 -O2 -g -DNDEBUG 标志

#### 2 设置优化级别的方式

- cmake-gui
- 终端命令参数`DCMAKE_BUILD_TYPE`

```terminal
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### G-compile-flags

#### 1 设置每个目标的c++标志

- `target_compile_definitions()`：预处理器的定义，这将填充库的`INTERFACE_COMPILE_DEFINITIONS`并根据范围将定义推送到链接的目标

```cmake
target_compile_definitions(cmake_examples_compile_flags
    PRIVATE EX3
)
```

- `target_compile_options()`：编译选项

#### 2 设置默认的c++标志

- `CMAKE_CXX_FLAGS`：c++标志

```cmake
# 将现有的c++ flags拼接上“-EX2”
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEX2" CACHE STRING "Set C++ Compiler Flags" FORCE)
```

- `CMAKE_C_FLAGS`：c标志
- `CMAKE_LINKER_FLAGS`：链接器标志

> 设置 `CMAKE_C_FLAGS` 和 `CMAKE_CXX_FLAGS` 后，将为该目录或任何包含的子目录中的所有目标全局设置编译器标志/定义。 现在不建议一般使用此方法，首选 `target_compile_definitions` 函数。

#### 3 设置CMake的flag的其他方式

- cmake-gui
- terminal

```terminal
cmake .. -DCMAKE_CXX_FLAGS="-DEX3"
```

### H-third-party-library

#### 1 查找第三方库

- `find_package()`：会在`CMAKE_MODULE_PATH`的路径中寻找形式为`FindXXX.cmake`的文件

```cmake
find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)
```

对于以上例子的解释如下：

- Boost - 库的名称。这是用于查找模块文件 FindBoost.cmake 的一部分。

- 1.46.1 - 要查找的 Boost 的最低版本

- REQUIRED - 告诉模块这是必需的，如果找不到则失败

- COMPONENTS - 要查找的库的列表。

#### 2 检查第三方库是否找到

大多数被包含的软件包会设置一个名为 `XXX_FOUND` 的变量，可以用来检查该软件包是否在系统上可用。

```cmake
if(Boost_FOUND)
    message ("boost found")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()
```

#### 3 导出变量

在找到一个软件包之后，通常会导出一些变量，用于通知用户在哪里找到库、头文件或可执行文件。与 `XXX_FOUND`变量类似，这些变量是特定于软件包的，并且通常在`FindXXX.cmake`文件的顶部有文档说明。

在这个示例中导出的变量包括：

| 变量名                   | 含义                        |
| ------------------------ | --------------------------- |
| Boost_INCLUDE_DIRS       | 指向 Boost 头文件的路径     |
| Boost_SYSTEM_LIBRARY     | 指向 Boost 系统库的路径     |
| Boost_FILESYSTEM_LIBRARY | 指向 Boost 文件系统库的路径 |

#### 4 别名变量

对于 Boost，你可以用以下方式替换这个示例中的内容：

- 将 `Boost_INCLUDE_DIRS` 替换为 `Boost::boost`，对于只有头文件的库（header only libraries）。
- 将 `Boost_FILESYSTEM_LIBRARY` 替换为 `Boost::filesystem`。
- 将 `Boost_SYSTEM_LIBRARY` 替换为 `Boost::system`。如果你包含了 `Boost::filesystem`，它会自动包含 `Boost::system`。

#### 5 <a name="third-party-example">示例完整的CMakeLists</a>

```cmake
cmake_minimum_required(VERSION 3.0)

# Set the project name
project (third_party_include)

# find a boost install with the libraries filesystem and system
find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)

# check if boost was found
if(Boost_FOUND)
    message ("boost found")
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()

# Add an executable
add_executable(third_party_include main.cpp)

# Include the boost headers
target_include_directories( third_party_include
    PRIVATE ${Boost_INCLUDE_DIRS}
)

# link against the boost libraries
target_link_libraries( third_party_include
    PRIVATE
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)
```

### I-compling-with-clang

#### 1 使用clang进行编译的方法

- cmake-gui
- terminal

```terminal
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

#### 2 示例

```terminal
$ mkdir build.clang

$ cd build.clang/

$ cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

### J-building-with-ninja

#### 1 调用Ninja生成器

```terminal
$ cmake .. -G Ninja

$ ls
build.ninja  CMakeCache.txt  CMakeFiles  cmake_install.cmake  rules.ninja
```

在执行上述步骤后，CMake 将会生成所需的 Ninja 构建文件，然后可以使用 ninja 命令来运行这些构建文件。

#### 2 使用Ninja构建项目的完整过程

- `ninja -v`命令：`ninja -v` 是用于执行 Ninja 构建系统的命令，其中 `-v` 选项表示 "verbose"，即详细模式。在详细模式下，Ninja 将会输出更多的构建信息，包括每个步骤的命令和输出。

```terminal
$ mkdir build.ninja

$ cd build.ninja/

$ cmake .. -G Ninja

$ ninja -v

$ ls
build.ninja  CMakeCache.txt  CMakeFiles  cmake_install.cmake  hello_cmake  rules.ninja

$ ./hello_cmake
Hello CMake!
```

### K-imported-targets

#### 1 使用导入别名目标的方式来链接第三方库

导入的目标是由 `FindXXX` 模块导出的只读目标。 导入目标的好处是它们还可以**填充包含目录和链接库**。

```cmake
target_link_libraries( imported_targets
    PRIVATE
    Boost::filesystem
)
```

#### 2 与[third-party-library](#third-party-example)样例的对比

```cmake
# Include the boost headers
target_include_directories( third_party_include
    PRIVATE ${Boost_INCLUDE_DIRS}
)

# link against the boost libraries
target_link_libraries( third_party_include
    PRIVATE
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)
# ================================================

# link against the boost libraries
target_link_libraries( imported_targets
    PRIVATE
    Boost::filesystem
)
```

前者以**变量**的形式存储目录路径和库路径，后者使用了 CMake 中的**导入目标**（Imported Target）功能。导入目标是由 `find_package()` 命令通过 Find 模块生成的一种特殊的 CMake 目标，它封装了库的信息，包括库的位置、头文件路径、链接信息等，使用**导入目标**隐藏了具体的库路径和链接细节，使得链接更加简洁和可维护

### L-cpp-standard

#### 1 通常方法：通过`CMAKE_CXX_FLAGS`设置c++标准

- `CHECK_CXX_COMPILER_FLAG()`：

```cmake
# 告诉 CMake 包含此函数使其可用。
include(CheckCXXCompilerFlag)

# "-std=c++11"：这是一个字符串，表示要检查的编译器标志，COMPILER_SUPPORTS_CXX11：这是一个变量名，用于接收检查结果。在这个例子中，它是一个布尔类型的变量，表示编译器是否支持 C++11 标准。如果支持，则该变量值为 TRUE，否则为 FALSE。
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
```

- 检查

```cmake
if(COMPILER_SUPPORTS_CXX11)#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
```

#### 2 通过`CMAKE_CXX_STANDARD`设置c++标准

- `CMAKE_CXX_STANDARD`：设置`CMAKE_CXX_STANDARD`变量会导致所有目标的`CXX_STANDARD`属性设置，所以需要设置适当的标志。

```cmake
# set the C++ standard to C++ 11
set(CMAKE_CXX_STANDARD 11)
```

> `CMAKE_CXX_STANDARD` 变量会在没有失败的情况下回退到最接近的适当标准。例如，如果你请求 `-std=gnu11`，可能会得到 `-std=gnu0x`。
>
> 这可能导致编译时出现意外的失败。

#### 3 通过`CMAKE_CXX_COMPILE_FEATURES `变量

- `target_compile_features()`：在目标上调用`target_compile_features()`函数将查看传入的功能并确定用于您的目标的正确编译器标志。

```cmake
# set the C++ standard to the approriate standard for using auto
target_compile_features(hello_cpp11 PUBLIC cxx_auto_type)

# Print the list of known compile features for this version of CMake
message("List of compile features: ${CMAKE_CXX_COMPILE_FEATURES}")
```

## 02-sub-projects

### A-basic

**示例项目结构**

```terminal
$ tree
.
├── CMakeLists.txt
├── subbinary
│   ├── CMakeLists.txt
│   └── main.cpp
├── sublibrary1
│   ├── CMakeLists.txt
│   ├── include
│   │   └── sublib1
│   │       └── sublib1.h
│   └── src
│       └── sublib1.cpp
└── sublibrary2
    ├── CMakeLists.txt
    └── include
        └── sublib2
            └── sublib2.h
```

#### 1 添加子目录

- `add_subdirectory()`

```cmake
add_subdirectory(sublibrary1)
add_subdirectory(sublibrary2)
add_subdirectory(subbinary)
```

#### 2 引用子项目目录

当使用`project()`命令创建项目时，CMake将自动创建许多**变量**，这些变量可用于引用有关项目的详细信息。 这些变量可以被其他**子项目或主项目**使用。CMake创建的变量如下表所示：

| 变量               | 说明                                                         |
| ------------------ | :----------------------------------------------------------- |
| PROJECT_NAME       | 当前project()设置的项目名称。                                |
| CMAKE_PROJECT_NAME | 由project()命令设置的第一个项目的名称，即**顶层项目**。      |
| PROJECT_SOURCE_DIR | 当前项目的源文件目录。                                       |
| PROJECT_BINARY_DIR | 当前项目的构建目录。                                         |
| name_SOURCE_DIR    | 项目的源目录名为“name”。 在此示例中，创建的源目录将为 `sublibrary1_SOURCE_DIR`、`sublibrary2_SOURCE_DIR` 和 `subbinary_SOURCE_DIR`。 |
| name_BINARY_DIR    | 项目的二进制目录名为“name”。 在此示例中，创建的二进制目录为 `sublibrary1_BINARY_DIR`、`sublibrary2_BINARY_DIR` 和 `subbinary_BINARY_DIR`。 |

#### 3 仅含头文件的库

如果你有一个作为仅包含头文件的库，CMake 支持使用`INTERFACE`目标来创建一个没有任何构建输出的目标。

```cmake
add_library(${PROJECT_NAME} INTERFACE)
```

创建目标时，您还可以使用`INTERFACE`范围<a name="子项目包含目录">包含该目标的目录</a>。`INTERFACE`范围用于制定目标要求，这些目标要求可在链接该目标的任何库中使用，但不能在目标本身的编译中使用。

```cmake
target_include_directories(${PROJECT_NAME}
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)
```

#### 4 从子项目引用库

如果子项目创建了库，则可以通过在`target_link_libraries()`命令中调用项目名称来引用该库。 这意味着**您不必引用新库的完整路径，它会作为依赖项添加**。

```cmake
target_link_libraries(subbinary
    PUBLIC
    sublibrary1
)
```

此外可以创建一个别名目标，它允许在**只读**上下文中引用该目标。

```cmake
# 创建库以及别名
add_library(sublibrary2)
add_library(sub::lib2 ALIAS sublibrary2)
```

```cmake
# 使用别名引用子库
target_link_libraries(subbinary
    sub::lib2
)
```

#### 5 从子项目引用库目录

从子项目添加库时，从 cmake v3 开始，无需在使用它们的二进制文件的**包含目录（include）**中添加**项目包含目录**。

这是由创建库时[`target_include_directories()`](#子项目包含目录)命令中的范围控制的。 在此示例中，由于子二进制可执行文件链接了 sublibrary1 和 sublibrary2 库，因此在使用库的`PUBLIC`和`INTERFACE`范围导出时，它将自动包含 `${sublibrary1_SOURCE_DIR}/include`和 `${sublibrary2_SOURCE_DIR}/include`文件夹。

## 03-code-generation

### A-configure-files

在调用 cmake 过程中，可以创建使用 CMakeLists.txt 和 CMake 缓存的变量的文件。在 CMake 生成过程中，该文件会被复制到一个新的位置，并且任何 cmake 变量都会被替换。

#### 1 添加配置文件

- `config_file()`：参数有两个，第一个参数为源文件，第二个参数为目标文件

```cmake
configure_file(ver.h.in ${PROJECT_BINARY_DIR}/ver.h)

configure_file(path.h.in ${PROJECT_BINARY_DIR}/path.h @ONLY)
```

第一个例子允许使用`${}`和`@@`来定义CMake变量，第二个例子仅允许使用`@@`

`ver.h.in`和`path.h.in`内容如下：

- `ver.h.in`

```cpp
#ifndef __VER_H__
#define __VER_H__

// version variable that will be substituted by cmake
// This shows an example using the $ variable type
const char* ver = "${cf_example_VERSION}";

#endif
```

- `path.h.in`

```cpp
#ifndef __PATH_H__
#define __PATH_H__

// version variable that will be substituted by cmake
// This shows an example using the @ variable type
const char* path = "@CMAKE_SOURCE_DIR@";

#endif
```

在运行`config_file()`函数后，在`build`文件夹中会生成`ver.h`，`path.h`文件，区别在于其中的表达式会被换成CMake变量的真实值，如下所示：

- `ver.h`

```cpp
#ifndef __VER_H__
#define __VER_H__

// version variable that will be substituted by cmake
// This shows an example using the $ variable type
const char* ver = "0.2.1";

#endif
```

- `path.h`

```cpp
#ifndef __PATH_H__
#define __PATH_H__

// version variable that will be substituted by cmake
// This shows an example using the @ variable type
const char* path = "/home/dingqin/Desktop/CMAKE/cmake-examples-3.0-minimum/03-code-generation/configure-files";

#endif
```

### B-protobuf

Protocol Buffers 是 Google 的一种数据序列化格式。 用户提供带有数据描述的`.proto`文件。 然后使用`protobuf`编译器，可以将proto文件翻译成包括C++在内的多种语言的源代码。

#### 1 安装protobuf编译器以及protobuf开发库

为了使用Protocol Buffers，首先需要安装：

```terminal
sudo apt-get install protobuf-compiler libprotobuf-dev
```

CMake protobuf 软件包导出并在此示例中使用的变量包括：

| 变量名                | 含义                          |
| --------------------- | ----------------------------- |
| PROTOBUF_FOUND        | 如果已安装 Protocol Buffers   |
| PROTOBUF_INCLUDE_DIRS | Protocol Buffers 的头文件路径 |
| PROTOBUF_LIBRARIES    | Protocol Buffers 的库文件路径 |

还有更多的变量被定义，并且可以通过查阅您的 `FindProtobuf.cmake` 文件顶部的文档找到

#### 2 生成源代码

Protobuf 的 CMake 软件包包含了许多辅助函数，以使代码生成更加简便。在此示例中，我们正在生成 C++ 源代码，使用了以下代码：

```cmake
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS AddressBook.proto)
```

其中：

- PROTO_SRCS - 用于存储 .pb.cc 文件的变量名称

- PROTO_HDRS - 用于存储 .pb.h 文件的变量名称

- AddressBook.proto - 要从中生成代码的 .proto 文件

#### 3 生成文件

在调用 PROTOBUF_GENERATE_CPP 函数之后，您将可以使用上述提到的变量。这些变量将被标记为一个自定义命令的输出，该命令调用 Protocol Buffers 编译器二进制文件来生成它们。

然后，为了让生成的文件生效，您应该将它们添加到一个库或可执行文件中。例如：

```cmake
add_executable(protobuf_example
    main.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS})
```


这将导致在调用该可执行文件的目标上使用 make 命令时，会调用 Protocol Buffers 编译器。

**当对 .proto 文件进行更改时，相关的源文件将被重新自动生成。**然而，如果对 .proto 文件没有进行更改，而您重新运行 make 命令，那么将不会执行任何操作。

## 04-static-analysis

### A-cppcheck-analysis

#### 1 案例简介

这个示例展示了如何调用 CppCheck 工具进行静态分析。它演示了如何为仓库中的每个项目创建一个分析目标。

其中包括以下代码：

1. 查找 cppcheck 可执行文件。
2. 为每个子项目添加 cppcheck 分析目标。
3. 生成一个整体的 make 分析目标，以对所有子项目进行静态分析。

这个示例中包含的文件有：

```terminal
$ tree
.
├── cmake
│   ├── analysis.cmake
│   └── modules
│       └── FindCppCheck.cmake
├── CMakeLists.txt
├── subproject1
│   ├── CMakeLists.txt
│   └── main1.cpp
└── subproject2
    ├── CMakeLists.txt
    └── main2.cpp
```

- `CMakeLists.txt` - 顶层 CMakeLists.txt 文件

- `cmake/analysis.cmake` - 包含添加分析目标函数的文件

- `cmake/modules/FindCppCheck.cmake` - 一个自定义的包模块，用于查找 CppCheck

- `subproject1/CMakeLists.txt` - 子项目1的 CMake 命令

- `subproject1/main.cpp` - 一个没有错误的子项目的源代码

- `subproject2/CMakeLists.txt` - 子项目2的 CMake 命令

- `subproject2/main2.cpp` - 一个包含错误的子项目的源代码

#### 2 安装cppcheck工具

```terminal
sudo apt-get install cppcheck
```

#### 3 概念

##### 1 添加自定义的包模块

自定义模块可用于查找程序、库和头文件，以包含在您的程序中。

##### 2 添加一个自定义模块

`cmake/modules/FindCppCheck.cmake` 文件包含了初始化一个自定义包模块的代码。

**以下是文件的拆解说明：**

在路径中搜索 cppcheck 二进制文件。一旦找到，将结果存储在 `CPPCHECK_BIN` 变量中。

```cmake
find_program(CPPCHECK_BIN NAMES cppcheck)
```

设置一些自定义的参数，稍后可以传递给 cppcheck。

```cmake
# 自定义参数：使用 4 个线程或核心来进行多线程处理

set (CPPCHECK_THREADS "-j 4" CACHE STRING "The -j argument to have cppcheck use multiple threads / cores")
```

设置要传递给 cppcheck 的参数。如果设置了该变量，将覆盖 `CPPCHECK_THREADS`。

```cmake
set (CPPCHECK_ARG "${CPPCHECK_THREADS}" CACHE STRING "The arguments to pass to cppcheck. If set will overwrite CPPCHECK_THREADS")
```

使用标准参数处理函数初始化查找包。这个函数将判断查找是否成功，并将结果传递给其他函数。其中`CPPCHECK`是一个用户定义的包名称，用于标识正在查找的包，`DEFAULT_MSG`是一个默认的查找失败时显示的消息。

```cmake
# 使用FindPackageHandleStandardArgs函数将查找的结果和相关的标准参数传递给其他函数

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    CPPCHECK
    DEFAULT_MSG
    CPPCHECK_BIN
    CPPCHECK_THREADS
    CPPCHECK_ARG)
```

将这些变量标记为高级选项，以便在 ccmake / cmake-gui 中可见并在缓存中设置。默认情况下，除非设置了 "显示高级选项" 标志，否则这些选项将不可见。

```cmake
mark_as_advanced(
    CPPCHECK_BIN
    CPPCHECK_THREADS
    CPPCHECK_ARG)
```

##### 3 设置自定义模块的路径

CMake 默认搜索模块的路径是 `/usr/share/cmake/Modules`。如果要包含自定义模块，您必须告诉 CMake 在哪里搜索它们。这可以通过变量 `${CMAKE_MODULE_PATH}` 来完成，该变量包含 CMake 将搜索模块的路径。

```cmake
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules
                      ${CMAKE_MODULE_PATH})
```

然后，要将包模块添加到您的 CMakeLists.txt 中，您可以调用

```cmake
find_package(CppCheck)
```

