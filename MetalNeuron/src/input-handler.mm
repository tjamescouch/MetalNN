#import "input-handler.h"
#import <GameController/GameController.h>
#import <AppKit/AppKit.h>

static KeyboardCallback globalCallback = nullptr;

@implementation InputHandler

+ (void)startKeyboardMonitoringWithCallback:(KeyboardCallback)callback {
    globalCallback = callback;
    
    if (@available(macOS 10.15, *)) {
        printf("[DEBUG] Initializing keyboard monitoring...\n");

        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(keyboardDidConnect:)
                                                     name:GCKeyboardDidConnectNotification
                                                   object:nil];

        if ([GCKeyboard coalescedKeyboard]) {
            printf("[DEBUG] GCKeyboard already connected.\n");
            [self setupKeyboard:[GCKeyboard coalescedKeyboard]];
        } else {
            printf("[DEBUG] No GCKeyboard available at startup. Waiting for connection...\n");
        }
    } else {
        printf("[ERROR] GCKeyboard not supported on this OS version.\n");
    }
}

+ (void)keyboardDidConnect:(NSNotification *)notification {
    printf("[DEBUG] GCKeyboard connected!\n");
    GCKeyboard *keyboard = notification.object;
    [self setupKeyboard:keyboard];
}

+ (void)setupKeyboard:(GCKeyboard *)keyboard {
    if (keyboard && keyboard.keyboardInput) {
        printf("[DEBUG] Setting up keyboard input handler...\n");

        keyboard.keyboardInput.keyChangedHandler = ^(GCKeyboardInput *keyboardInput,
                                                     GCControllerButtonInput *key,
                                                     GCKeyCode keyCode,
                                                     BOOL pressed) {
            if (globalCallback) {
                KeyPress kp = { keyCode, pressed };
                globalCallback(&kp);
            }

            // Suppress the system beep by consuming the key event
            [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown handler:^NSEvent *(NSEvent *event) {
                return nil; // Returning nil prevents event from propagating (stopping the beep)
            }];
        };
    } else {
        printf("[ERROR] keyboardInput is nil. No input events will be captured.\n");
    }
}

@end
