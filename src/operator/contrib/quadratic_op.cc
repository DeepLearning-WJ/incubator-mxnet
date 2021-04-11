/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file quadratic_op.cc
 * \brief CPU Implementation of quadratic op
 */
#include "./quadratic_op-inl.h"

namespace mxnet {
namespace op {

// 注册到CPU
DMLC_REGISTER_PARAMETER(QuadraticParam);// 注册QuadraticParam

NNVM_REGISTER_OP(_contrib_quadratic)
.describe(R"code(This operators implements the quadratic function.

.. math::
    f(x) = ax^2+bx+c

where :math:`x` is an input tensor and all operations
in the function are element-wise.

Example::

  x = [[1, 2], [3, 4]]
  y = quadratic(data=x, a=1, b=2, c=3)
  y = [[6, 11], [18, 27]]

The storage type of ``quadratic`` output depends on storage types of inputs
  - quadratic(csr, a, b, 0) = csr
  - quadratic(default, a, b, c) = default

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<QuadraticParam>)// 参数结构解析器
.set_num_inputs(1)// input 数量
.set_num_outputs(1)// output 数量
.set_attr<nnvm::FListInputNames>("FListInputNames",// 该函数用于添加在创建符号操作符时没有指定的缺失参数
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuadraticOpShape)
.set_attr<nnvm::FInferType>("FInferType", QuadraticOpType)
.set_attr<FInferStorageType>("FInferStorageType", QuadraticOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_contrib_backward_quadratic"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",// 哪个输出张量可以重用哪个输入张量的存储空间，不是为输出分配一个新的存储空间
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(QuadraticParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_backward_quadratic)// 下划线前缀表明这是一个未向用户公开的操作符
.set_attr_parser(ParamParser<QuadraticParam>)// ParamParser for backward
.set_num_inputs(2)// 反向传播为什么是两个?
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)// 确定图中的节点是向前节点还是向后节点
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpBackward<cpu>)// 注册backward
.set_attr<FComputeEx>("FComputeEx<cpu>", QuadraticOpForwardEx<cpu>);// 注册forward

}  // namespace op
}  // namespace mxnet
