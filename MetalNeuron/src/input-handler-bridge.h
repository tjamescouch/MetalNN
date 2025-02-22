#pragma once

#ifdef __cplusplus
#include <functional>
#include <string>
#import "key-press.h"

// Define a C++ callback type for handling keyboard events
using KeyboardEventCallback = std::function<void(const KeyPress&)>;

// Expose a function to start keyboard monitoring from C++
extern "C" void StartKeyboardMonitoring(KeyboardEventCallback callback);

#endif  // __cplusplus
