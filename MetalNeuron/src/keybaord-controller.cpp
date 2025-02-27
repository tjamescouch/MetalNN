//
//  keybaord-controller.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-26.
//

#include "keyboard-controller.h"

KeyboardController::KeyboardController() {
}

KeyboardController::~KeyboardController() {
}

void KeyboardController::keyPress(KeyPress* kp) {
    if (kp->pressed) {
        keyState[kp->code] = true;
    } else {
        keyState.erase(kp->code);
    }
}

void KeyboardController::handleKeyStateChange() {
    // Check for key code 9 ('F') to trigger the forward pass.
    auto it = keyState.find(9);
    if (it != keyState.end() && it->second && forwardCallback) {
        forwardCallback();
    }
    
    // Check for key code 15 ('L') to trigger learning.
    it = keyState.find(15);
    if (it != keyState.end() && it->second && learnCallback) {
        learnCallback();
    }
    
    // Check for key code 6 ('C') to trigger clearing.
    it = keyState.find(6);
    if (it != keyState.end() && it->second && clearCallback) {
        clearCallback();
    }
}

void KeyboardController::setForwardCallback(std::function<void()> callback) {
    forwardCallback = callback;
}

void KeyboardController::setLearnCallback(std::function<void()> callback) {
    learnCallback = callback;
}

void KeyboardController::setClearCallback(std::function<void()> callback) {
    clearCallback = callback;
}
