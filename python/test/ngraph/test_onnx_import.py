# ******************************************************************************
# Copyright 2018 Intel Corporation
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

import os
import numpy as np
import pytest

import ngraph as ng
from ngraph.impl.onnx_import import onnx_import
from test.ngraph.util import get_runtime

def test_import_onnx_function():
    runtime = get_runtime()
    dtype = np.float32
    shape = [1]
    parameter_a = ng.parameter(shape, dtype=dtype, name='A')
    parameter_b = ng.parameter(shape, dtype=dtype, name='B')
    parameter_c = ng.parameter(shape, dtype=dtype, name='C')
    cur_dir = os.path.dirname(__file__)
    model_path = os.path.join(cur_dir, '../../../test/models/onnx/add_abc.onnx')
    function = onnx_import.import_onnx_function_file(model_path)
    computation = runtime.computation_function(function, parameter_a, parameter_b, parameter_c)

    value_a = np.array([1.0], dtype=dtype)
    value_b = np.array([2.0], dtype=dtype)
    value_c = np.array([3.0], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([6], dtype=dtype))
