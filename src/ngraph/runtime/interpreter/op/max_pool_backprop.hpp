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

#include "ngraph/op/max_pool.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class MaxPoolBackpropExec;
        }
    }
}

class ngraph::runtime::interpreter::MaxPoolBackpropExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        return std::static_pointer_cast<ExecNode>(std::make_shared<MaxPoolBackpropExec>(node));
    }

    MaxPoolBackpropExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::MaxPoolBackprop>(node)}
    {
        (void)m_node; // Silence compiler warning
    }

    virtual ~MaxPoolBackpropExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        op::MaxPoolBackprop* max_pool_backprop = dynamic_cast<op::MaxPoolBackprop*>(&node);

        reference::max_pool_backprop<T>(args[0]->get_data_ptr<T>(),
                                        args[1]->get_data_ptr<T>(),
                                        out[0]->get_data_ptr<T>(),
                                        args[1]->get_shape(),
                                        out[0]->get_shape(),
                                        max_pool_backprop->get_window_shape(),
                                        max_pool_backprop->get_window_movement_strides(),
                                        max_pool_backprop->get_padding_below(),
                                        max_pool_backprop->get_padding_above());
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::MaxPoolBackprop_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::MaxPool> m_node;
};
