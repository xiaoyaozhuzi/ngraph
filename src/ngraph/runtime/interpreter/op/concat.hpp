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

#include "ngraph/op/concat.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/concat.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class ConcatExec;
        }
    }
}

class ngraph::runtime::interpreter::ConcatExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        return std::static_pointer_cast<ExecNode>(std::make_shared<ConcatExec>(node));
    }

    ConcatExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::Concat>(node)}
    {
        (void)m_node; // Silence compiler warning
    }

    virtual ~ConcatExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        std::vector<const T*> in_args;
        std::vector<Shape> in_shapes;
        for (std::shared_ptr<HostTensorView> arg : args)
        {
            in_args.push_back(arg->get_data_ptr<T>());
            in_shapes.push_back(arg->get_shape());
        }
        reference::concat<T>(in_args,
                             out[0]->get_data_ptr<T>(),
                             in_shapes,
                             out[0]->get_shape(),
                             m_node->get_concatenation_axis());
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::Concat_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::Concat> m_node;
};
