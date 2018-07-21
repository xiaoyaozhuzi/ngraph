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

#include <iostream>
#include <memory>
#include <vector>

#include "ngraph/op/abs.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/abs.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class AbsExec;
        }
    }
}

class ngraph::runtime::interpreter::AbsExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        std::cout << "create Abs" << std::endl;
        return std::static_pointer_cast<ExecNode>(std::make_shared<AbsExec>(node));
    }

    AbsExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::Abs>(node)}
    {
        (void)m_node; // Silence compiler warning

        std::cout << "Abs ctor" << std::endl;
    }

    virtual ~AbsExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        std::cout << "execute Abs" << std::endl;
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::Abs_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::Abs> m_node;
};
