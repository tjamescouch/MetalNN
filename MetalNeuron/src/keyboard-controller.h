//
//  keyboard-controller.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-26.
//

#ifndef KEYBOARD_CONTROLLER_H
#define KEYBOARD_CONTROLLER_H

#pragma once

#include "key-press.h"
#include <map>
#include <functional>

class KeyboardController {
public:
    KeyboardController();
    ~KeyboardController();
    
    // Process a key press event.
    void keyPress(KeyPress* kp);
    
    // Process current key states and invoke corresponding actions.
    void handleKeyStateChange();
    
    // Register callbacks for specific key actions.
    void setForwardCallback(std::function<void()> callback);
    void setLearnCallback(std::function<void()> callback);
    void setClearCallback(std::function<void()> callback);
    void setSaveCallback(std::function<void()> callback);
    void setLoadCallback(std::function<void()> callback);
    
private:
    std::map<long, bool> keyState;
    std::function<void()> forwardCallback;
    std::function<void()> learnCallback;
    std::function<void()> clearCallback;
    std::function<void()> saveCallback;
    std::function<void()> loadCallback;
};

#endif // KEYBOARD_CONTROLLER_H
