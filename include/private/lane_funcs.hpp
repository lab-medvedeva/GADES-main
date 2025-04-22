#pragma once

#include "hwy/base.h"

namespace lane
{

static size_t RoundDownTo(size_t val, size_t lanes)
{
  return val & (~(lanes - 1));
}

static size_t RoundUpTo(size_t val, size_t lanes) {
  return RoundDownTo(val + (lanes - 1), lanes);
}

static size_t Mod(size_t val, size_t lanes) {
  return val & (lanes - 1);
}

static size_t getPow(size_t lanes){
  return hwy::PopCount(lanes - 1);
}

} // namespace lane
