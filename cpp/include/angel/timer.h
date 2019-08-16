/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
//
// Created by leleyu on 2019-05-16.
//

#ifndef TORCH_ON_ANGEL_TIMER_H
#define TORCH_ON_ANGEL_TIMER_H

#include <chrono>

#define TIMER_START(_X) \
  auto _X##_start = std::chrono::steady_clock::now(), _X##_stop = _X##_start
#define TIMER_STOP(_X)  \
  _X##_stop = std::chrono::steady_clock::now()
#define TIMER_NSEC(_X) \
  std::chrono::duration_cast<std::chrono::nanoseconds>(_X##_stop - _X##_start).count()
#define TIMER_USEC(_X) \
  std::chrono::duration_cast<std::chrono::microseconds>(_X##_stop - _X##_start).count()
#define TIMER_MSEC(_X) \
  0.000001 * TIMER_NSEC(_X)
#define TIMER_SEC(_X) \
  0.000001 * TIMER_USEC(_X)


#endif //TORCH_ON_ANGEL_TIMER_H
