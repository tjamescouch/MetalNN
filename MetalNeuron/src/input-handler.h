//
//  input-handler.h
//  LearnMetalCPP
//
//  Created by James Couch on 2025-02-02.
//  Copyright © 2025 Apple. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <GameController/GameController.h>
#import "key-press.h"

// Define a callback block type that takes an NSString.
typedef void (^KeyboardCallback)(KeyPress *key);

@interface InputHandler : NSObject
+ (void)startKeyboardMonitoringWithCallback:(KeyboardCallback)callback;
@end
