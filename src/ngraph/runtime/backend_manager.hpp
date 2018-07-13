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

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ngraph
{
    namespace runtime
    {
        class Backend;

        /// @brief Interface to a generic backend.
        ///
        /// Backends are responsible for function execution and value allocation.
        class BackendManager
        {
            friend class Backend;

        private:
            /// @brief Create a new Backend object
            /// @param type The name of a registered backend, such as "CPU" or "GPU".
            ///   To select a subdevice use "GPU:N" where s`N` is the subdevice number.
            /// @returns shared_ptr to a new Backend or nullptr if the named backend
            ///   does not exist.
            static std::shared_ptr<Backend> create(const std::string& type);

            /// @brief Query the list of registered devices
            /// @returns A vector of all registered devices.
            static std::vector<std::string> get_registered_devices();

            static bool register_backend(const std::string& name, std::shared_ptr<Backend>);

            static void* open_shared_library(std::string type);
            static std::map<std::string, std::string> get_registered_device_map();
        };
    }
}
