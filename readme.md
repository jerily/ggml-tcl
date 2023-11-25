# ggml-tcl

TCL bindings for [ggml](https://github.com/ggerganov/ggml).

**Note that this project is under active development.**

## Build Dependency

Download latest stable version on Linux:
```bash
git clone https://github.com/ggerganov/ggml.git
cd ggml
mkdir build
cd build
cmake .. \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF
make
make install
```

## Installation

```bash
git clone https://github.com/jerily/ggml-tcl.git
cd ggml-tcl
mkdir build
cd build
# change "TCL_LIBRARY_DIR" and "TCL_INCLUDE_DIR" to the correct paths
cmake .. \
  -DTCL_LIBRARY_DIR=/usr/local/lib \
  -DTCL_INCLUDE_DIR=/usr/local/include
make
make install
```