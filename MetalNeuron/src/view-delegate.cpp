//
//  view-delegate.cpp
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//

#include "view-delegate.h"
#include "model-config.h"
#include <iostream>

const char* defaultModelFilePath = "model-config.yml";

#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate(MTL::Device* pDevice)
: MTK::ViewDelegate()
, _pDevice(pDevice)
, _pComputer(nullptr)
{
    loadModelFromFile(defaultModelFilePath);
}

ViewDelegate::~ViewDelegate()
{
    delete _pComputer;
}

bool ViewDelegate::loadModelFromFile(const std::string& filePath)
{
    try {
        ModelConfig config = ModelConfig::loadFromFile(filePath);

        if (_pComputer != nullptr) {
            delete _pComputer; // Clean up existing engine, if any
        }

        // Instantiate NeuralEngine using your existing constructor
        _pComputer = new NeuralEngine(_pDevice, 10, config);

        std::cout << "✅ NeuralEngine loaded with model: " << config.name << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error loading model: " << e.what() << std::endl;
        return false;
    }
}

void ViewDelegate::drawInMTKView(MTK::View* pView)
{
    pView->setDepthStencilPixelFormat(MTL::PixelFormatDepth32Float);
    pView->setClearDepth(1.0);

    if (_pComputer) {
        _pComputer->runInference(); // or your actual method call
    }
}

void ViewDelegate::drawableSizeWillChange(MTK::View* pView, CGSize size)
{
    // Handle resize events if needed
}

NeuralEngine* ViewDelegate::getComputer()
{
    return _pComputer;
}

#pragma endregion ViewDelegate }
