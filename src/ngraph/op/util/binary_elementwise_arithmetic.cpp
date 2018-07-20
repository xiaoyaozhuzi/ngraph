/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

using namespace std;
using namespace ngraph;

op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(
    const std::string& node_type,
    const std::shared_ptr<Node>& arg0,
    const std::shared_ptr<Node>& arg1)
    : BinaryElementwise(node_type, arg0->get_element_type(), arg0, arg1)
{
}

void op::util::BinaryElementwiseArithmetic::validate_and_infer_types()
{
    BinaryElementwise::validate_and_infer_types();
    if (get_input_element_type(0) != get_input_element_type(1))
    {
        throw ngraph_error("Arguments must have the same tensor view element type");
    }

    if (get_input_element_type(0) == element::boolean)
    {
        throw ngraph_error("Operands for arithmetic operators must have numeric element type");
    }
}
