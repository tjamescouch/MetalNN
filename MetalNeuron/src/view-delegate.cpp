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
#include "mnist-data-loader.h"

const char* defaultModelFilePath = "ocr.yml";

#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate(MTL::Device* pDevice)
: MTK::ViewDelegate()
, _pDevice(pDevice)
, _pComputer(nullptr)
, _pDataManager(nullptr)
{
    ModelConfig config = ModelConfig::loadFromFile(getDefaultModelFilePath());

    // Instantiate DataManager first with dataset from config
    Dataset* dataset = nullptr;
    if (config.dataset.type == "mnist") {
        dataset = new MNISTDataLoader(
            config.dataset.images,
            config.dataset.labels
        );
    } else {
        throw std::runtime_error("Unsupported dataset type");
    }

    _pDataManager = new DataManager(dataset, config.first_layer_time_steps);

    // Instantiate NeuralEngine using the updated constructor with DataManager
    _pComputer = new NeuralEngine(_pDevice, config, _pDataManager);

    std::cout << "✅ NeuralEngine loaded with model: " << config.name << std::endl;
}

ViewDelegate::~ViewDelegate()
{
    delete _pComputer;
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
    fs::path resourcePath = executablePath.parent_path().parent_path() / "Resources" / defaultModelFilePath;

    if (!fs::exists(resourcePath)) {
        throw std::runtime_error("❌ Could not find confiuration yml at " + resourcePath.string());
    }
    std::cout << "📂 Loaded file " << defaultModelFilePath << std::endl;

    return resourcePath.string();
}

#pragma endregion ViewDelegate }
