#pragma once

namespace lane
{

static size_t RoundDownTo(size_t val, size_t lanes)
{
  return val & (~(lanes - 1));
}

static size_t RoundUpTo(size_t val, size_t lanes) {
  return RoundDownTo(val + (lanes - 1), lanes);
}

} // namespace lane
