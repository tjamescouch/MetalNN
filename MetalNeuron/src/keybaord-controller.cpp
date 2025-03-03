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
    // F
    auto it = keyState.find(9);
    if (it != keyState.end() && it->second && forwardCallback) {
        forwardCallback();
    }
    
    // L
    it = keyState.find(15);
    if (it != keyState.end() && it->second && learnCallback) {
        learnCallback();
    }
    
    // C
    it = keyState.find(6);
    if (it != keyState.end() && it->second && clearCallback) {
        clearCallback();
    }
    
    // S
    it = keyState.find(22);
    if (it != keyState.end() && it->second && clearCallback) {
        saveCallback();
    }
    
    // O
    it = keyState.find(18);
    if (it != keyState.end() && it->second && clearCallback) {
        loadCallback();
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

void KeyboardController::setSaveCallback(std::function<void()> callback) {
    saveCallback = callback;
}

void KeyboardController::setLoadCallback(std::function<void()> callback) {
    loadCallback = callback;
}
