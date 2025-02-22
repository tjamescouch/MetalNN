// input-handler-bridge.mm
#import "input-handler-bridge.h"
#import "input-handler.h"  // Include the Objective-C keyboard handler

// Static C++ callback variable to store the function pointer
static KeyboardEventCallback cppKeyboardCallback = nullptr;



// Ensure InputHandlerBridge inherits from NSObject
@interface InputHandlerBridge : NSObject
@end

@implementation InputHandlerBridge

+ (void)startKeyboardMonitoringWithCppCallback:(KeyboardEventCallback)callback {
    // Store the C++ callback for use inside the block
    cppKeyboardCallback = callback;

    // Call the Objective-C method and forward the event to C++
    [InputHandler startKeyboardMonitoringWithCallback:^(KeyPress* kp) {
        if (cppKeyboardCallback) {
            cppKeyboardCallback(*kp);
        }
    }];
}

@end

// Expose the function to C++
void StartKeyboardMonitoring(KeyboardEventCallback callback) {
    [InputHandlerBridge startKeyboardMonitoringWithCppCallback:callback];
}
