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
#include "mnist-dataset.h"
#include "function-dataset.h"
#include "math-lib.h"
#include "configuration-manager.h"

//const char* modelFilename = "ocr.yml";
//const char* modelFilename = "simple-ocr.yml";
//const char* modelFilename = "ocr-with-dropout.yml";
//const char* modelFilename = "ocr-with-batch-normalization.yml";
//const char* modelFilename = "ocr-complete.yml";
const char* modelFilename = "feed-forward.yml";
//const char* modelFilename = "residual-connection.yml";
//const char* modelFilename = "gelu.yml";
//const char* modelFilename = "multi-dense-layer.yml";
//const char* modelFilename = "single-dense-layer.yml";

#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate(MTL::Device* pDevice)
: MTK::ViewDelegate()
, _pDevice(pDevice)
, _pComputer(nullptr)
, _pDataManager(nullptr)
{
    static ModelConfig config = ModelConfig::loadFromFile(getDefaultModelFilePath());
    config.filename = modelFilename;
    
    ConfigurationManager::instance().setConfig(&config);

    // Instantiate DataManager first with dataset from config
    Dataset* dataset = nullptr;
    if (config.dataset.type == "mnist") {
        dataset = new MNISTDataset(
            config.dataset.images,
            config.dataset.labels
        );
    } else if (config.dataset.type == "function") {
        dataset = new FunctionDataset(mathlib::inputFunc, mathlib::targetFunc,
                                    config.layers.front().params.at("input_size").get_value<int>(),
                                    config.layers.back().params.at("output_size").get_value<int>(),
                                    1000); //Arbitrary dataset size
    } else {
        throw std::runtime_error("Unsupported dataset type: " + config.dataset.type);
    }

    _pDataManager = new DataManager(dataset);
    
    // Instantiate NeuralEngine using the updated constructor with DataManager
    _pComputer = new NeuralEngine(_pDevice, config, _pDataManager);

    Logger::log << "✅ NeuralEngine loaded with model: " << config.name << std::endl;
}

ViewDelegate::~ViewDelegate()
{
    delete _pComputer;
}

void ViewDelegate::drawInMTKView(MTK::View* pView)
{
    pView->setDepthStencilPixelFormat(MTL::PixelFormatDepth32Float);
    pView->setClearDepth(1.0);
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
    fs::path resourcePath = executablePath.parent_path().parent_path() / "Resources" / modelFilename;

    if (!fs::exists(resourcePath)) {
        throw std::runtime_error("❌ Could not find configuration yml at " + resourcePath.string());
    }
    Logger::log << "📂 Loaded file " << modelFilename << std::endl;

    return resourcePath.string();
}

#pragma endregion ViewDelegate }
