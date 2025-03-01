//
//  view-delegate.cpp
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//

#include "view-delegate.h"
#include "model-config.h"
#include <iostream>
#include <filesystem>
#include <mach-o/dyld.h>

const char* defaultModelFilePath = "model-config.yml";

#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate(MTL::Device* pDevice)
: MTK::ViewDelegate()
, _pDevice(pDevice)
, _pComputer(nullptr)
{
    loadModelFromFile(getDefaultModelFilePath());
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
        _pComputer = new NeuralEngine(_pDevice, config);

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

std::string ViewDelegate::getDefaultModelFilePath() {
    namespace fs = std::filesystem;

    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        throw std::runtime_error("❌ Executable path buffer too small.");
    }

    fs::path executablePath = fs::canonical(path);
    fs::path resourcePath = executablePath.parent_path().parent_path() / "Resources" / "model-config.yml";

    if (!fs::exists(resourcePath)) {
        throw std::runtime_error("❌ Could not find model-config.yml at " + resourcePath.string());
    }

    return resourcePath.string();
}

#pragma endregion ViewDelegate }
