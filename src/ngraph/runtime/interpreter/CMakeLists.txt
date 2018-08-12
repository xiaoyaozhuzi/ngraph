# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set (SRC
    op/abs.cpp
    op/acos.cpp
    op/add.cpp
    op/allreduce.cpp
    op/and.cpp
    op/asin.cpp
    op/atan.cpp
    op/avg_pool.cpp
    op/batch_norm.cpp
    op/broadcast.cpp
    op/ceiling.cpp
    op/concat.cpp
    op/constant.cpp
    op/convert.cpp
    op/convolution.cpp
    op/cos.cpp
    op/cosh.cpp
    op/divide.cpp
    op/dot.cpp
    op/equal.cpp
    op/exp.cpp
    op/floor.cpp
    op/function_call.cpp
    op/get_output_element.cpp
    op/greater.cpp
    op/greater_eq.cpp
    op/less.cpp
    op/less_eq.cpp
    op/log.cpp
    op/max.cpp
    op/maximum.cpp
    op/max_pool_backprop.cpp
    op/max_pool.cpp
    op/min.cpp
    op/minimum.cpp
    op/multiply.cpp
    op/negative.cpp
    op/not.cpp
    op/not_equal.cpp
    op/one_hot.cpp
    op/or.cpp
    op/pad.cpp
    op/parameter.cpp
    op/power.cpp
    op/product.cpp
    op/reduce.cpp
    op/reduce_window.cpp
    # op/relubackprop.cpp
    op/relu.cpp
    op/replace_slice.cpp
    op/reshape.cpp
    op/result.cpp
    op/reverse.cpp
    op/reverse_sequence.cpp
    op/select_and_scatter.cpp
    op/select.cpp
    op/sigmoid.cpp
    op/sign.cpp
    op/sin.cpp
    op/sinh.cpp
    op/slice.cpp
    op/softmax.cpp
    op/sqrt.cpp
    op/stop_gradient.cpp
    op/subtract.cpp
    op/sum.cpp
    op/tan.cpp
    op/tanh.cpp
)

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    add_definitions(-DNGRAPH_DISTRIBUTED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    link_libraries(${MPI_CXX_LIBRARIES})
endif()

if (NGRAPH_INTERPRETER_ENABLE)
    add_library(interpreter_backend SHARED int_backend.cpp exec_node.cpp ${SRC})
    set_target_properties(interpreter_backend PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    target_link_libraries(interpreter_backend PUBLIC ngraph)
    set_target_properties(interpreter_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS interpreter_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
