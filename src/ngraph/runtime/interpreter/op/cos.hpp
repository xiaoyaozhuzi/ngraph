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

#include "ngraph/op/cos.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/cos.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class CosExec;
        }
    }
}

class ngraph::runtime::interpreter::CosExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        return std::static_pointer_cast<ExecNode>(std::make_shared<CosExec>(node));
    }

    CosExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::Cos>(node)}
    {
        (void)m_node; // Silence compiler warning
    }

    virtual ~CosExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        reference::cos<T>(
            args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::Cos_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::Cos> m_node;
};
