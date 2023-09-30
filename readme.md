# ggml-tcl

TCL bindings for [ggml](https://github.com/ggerganov/ggml).

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