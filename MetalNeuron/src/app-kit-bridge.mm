//
//  app-kit-bridge.mm
//  MetalNeuron
//
//  Created by James Couch on 2025-03-12.
//

#import <Cocoa/Cocoa.h>
#include "app-kit-bridge.h"

static NSTextView* globalTextView = nil;

extern "C" void setupTextField(void* nsWindow) {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSWindow* window = (__bridge NSWindow*)nsWindow;

        NSRect frame = [window.contentView bounds];

        NSScrollView* scrollView = [[NSScrollView alloc] initWithFrame:frame];
        [scrollView setHasVerticalScroller:YES];
        [scrollView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];

        globalTextView = [[NSTextView alloc] initWithFrame:[[scrollView contentView] bounds]];
        [globalTextView setEditable:NO];
        [globalTextView setSelectable:YES];

        // Explicitly set text attributes:
        NSMutableParagraphStyle* paragraphStyle = [[NSMutableParagraphStyle alloc] init];
        [paragraphStyle setLineSpacing:0.5]; // Adds spacing between lines

        [globalTextView setTypingAttributes:@{
            NSFontAttributeName: [NSFont fontWithName:@"Menlo" size:11],
            NSParagraphStyleAttributeName: paragraphStyle,
            NSForegroundColorAttributeName: [NSColor textColor]
        }];

        [globalTextView setBackgroundColor:[NSColor textBackgroundColor]];
        [globalTextView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];

        [scrollView setDocumentView:globalTextView];
        [window.contentView addSubview:scrollView];
    });
}

extern "C" void updateTextField(const char* message) {
    if (globalTextView) {
        const char* safeMsg = message ? message : "";
        NSString* safeString = [NSString stringWithUTF8String:safeMsg];
        if (!safeString) safeString = @"";

        dispatch_async(dispatch_get_main_queue(), ^{
            NSMutableParagraphStyle* paragraphStyle = [[NSMutableParagraphStyle alloc] init];
            [paragraphStyle setLineSpacing:0.5];

            NSDictionary* attributes = @{
                NSForegroundColorAttributeName : [NSColor textColor],
                NSFontAttributeName : [NSFont fontWithName:@"Menlo" size:11],
                NSParagraphStyleAttributeName : paragraphStyle
            };

            NSAttributedString* attributedMessage = [[NSAttributedString alloc]
                initWithString:[safeString stringByAppendingString:@"\n"]
                attributes:attributes];

            NSScrollView* scrollView = [globalTextView enclosingScrollView];
            NSRect visibleRect = [scrollView contentView].documentVisibleRect;
            NSRect documentRect = [[scrollView documentView] bounds];

            BOOL isAtBottom = NSMaxY(visibleRect) >= NSMaxY(documentRect) - 1.0;

            [[globalTextView textStorage] appendAttributedString:attributedMessage];

            if (isAtBottom) {
                [globalTextView scrollRangeToVisible:NSMakeRange([[globalTextView string] length], 0)];
            }
        });
    }
}
