```

################################################################################
gitpod /workspace/ranking_bandits (abndrank) $
cd topkbandit
source ../init.sh
pyenv install 3.8.13 && pyenv global 3.8.13 && python --version
pip3 install gdown
/workspace/.pyenv_mirror/fakeroot/versions/3.8.13/bin/python3.8 -m pip install --upgrade pip
pip3 install gdown
pip3 install -r requirements.txt
export PYTHONPATH="$(pwd)"
pip install pybind11
python -c "import pybind11; print(pybind11.get_cmake_dir())"
python -c "import pybind11; print(pybind11)"
ln -s lib/pybind11         /workspace/.pyenv_mirror/user/current/lib/python3.8/site-packages/pybind11
cd ..
cmake eigen -DINCLUDE_INSTALL_DIR=/usr/local/include/ 
sudo make install
pip install -e .




###############################################################################
#### Create CMakeLists.txt

cmake_minimum_required(VERSION 2.8.12)

project(xcb)
# Set source directory
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_DIR "src/xcb")
list(APPEND CMAKE_PREFIX_PATH "/workspace/.pyenv_mirror/user/current/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR}/corelib)
set(SOURCES "${SOURCE_DIR}/corelib/toy.cpp" "${SOURCE_DIR}/corelib/xcb_inference.cpp")
include_directories(lib/eigen)
pybind11_add_module(core ${SOURCES} "${SOURCE_DIR}/corelib/bindings.cpp")





```