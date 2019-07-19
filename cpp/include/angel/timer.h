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
