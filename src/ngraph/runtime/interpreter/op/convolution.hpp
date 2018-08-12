/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the \"License\");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an \"AS IS\" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>

#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/convolution.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class ConvolutionExec;
        }
    }
}

class ngraph::runtime::interpreter::ConvolutionExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        return std::static_pointer_cast<ExecNode>(std::make_shared<ConvolutionExec>(node));
    }

    ConvolutionExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::Convolution>(node)}
    {
        (void)m_node; // Silence compiler warning
    }

    virtual ~ConvolutionExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        reference::convolution<T>(args[0]->get_data_ptr<T>(),
                                  args[1]->get_data_ptr<T>(),
                                  out[0]->get_data_ptr<T>(),
                                  args[0]->get_shape(),
                                  args[1]->get_shape(),
                                  out[0]->get_shape(),
                                  m_node->get_window_movement_strides(),
                                  m_node->get_window_dilation_strides(),
                                  m_node->get_padding_below(),
                                  m_node->get_padding_above(),
                                  m_node->get_data_dilation_strides(),
                                  0,
                                  1,
                                  1,
                                  0,
                                  0,
                                  1,
                                  false);
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::Convolution_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::Convolution> m_node;
};
