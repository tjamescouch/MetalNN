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
        [globalTextView setFont:[NSFont fontWithName:@"Menlo" size:12]];
        [globalTextView setTextColor:[NSColor systemRedColor]];
        [globalTextView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];

        // Ensure text color adapts automatically for dark/light mode:
        [globalTextView setBackgroundColor:[NSColor textBackgroundColor]];

        [scrollView setDocumentView:globalTextView];
        [window.contentView addSubview:scrollView];
    });
}

extern "C" void updateTextField(const char* message) {
    if (globalTextView && message) {
        NSString* safeString = [NSString stringWithUTF8String:message];

        dispatch_async(dispatch_get_main_queue(), ^{
            NSScrollView* scrollView = [globalTextView enclosingScrollView];
            NSClipView* clipView = [scrollView contentView];
            NSRect visibleRect = [clipView documentVisibleRect];
            NSRect documentRect = [[scrollView documentView] bounds];

            BOOL isAtBottom = NSMaxY(visibleRect) >= NSMaxY(documentRect) - NSHeight(visibleRect) * 0.05;

            NSString* appendString = [safeString hasSuffix:@"\n"] ? safeString : [safeString stringByAppendingString:@"\n"];
            NSDictionary* attributes = @{NSForegroundColorAttributeName : [NSColor textColor]};
            NSAttributedString* attributedMessage = [[NSAttributedString alloc] initWithString:appendString attributes:attributes];

            [[globalTextView textStorage] appendAttributedString:attributedMessage];

            if (isAtBottom) {
                [globalTextView scrollRangeToVisible:NSMakeRange([[globalTextView string] length], 0)];
            }
        });
    }
}
