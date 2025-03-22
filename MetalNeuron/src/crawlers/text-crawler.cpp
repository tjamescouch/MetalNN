//
//  text-crawler.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#include "text-crawler.h"
#include <fstream>
#include <sstream>
#include <cassert>

// Constructor explicitly initializes crawler parameters
TextCrawler::TextCrawler(const std::vector<std::string>& filepaths,
                         size_t sequenceLength,
                         size_t samplesPerFile)
    : filepaths_(filepaths),
      sequenceLength_(sequenceLength),
      samplesPerFile_(samplesPerFile),
      currentSampleCount_(0),
      generator_(std::random_device{}()) {
    assert(!filepaths_.empty() && "Filepaths cannot be empty.");
    loadNextFile();
}

TextCrawler::~TextCrawler() {
    // Destructor explicitly empty (RAII manages resources)
}

// Explicitly loads content of one file into memory
void TextCrawler::loadFile(const std::string& filepath) {
    std::ifstream file(filepath);
    assert(file.is_open() && "Failed to open file.");

    std::stringstream buffer;
    buffer << file.rdbuf();
    currentFileContent_ = buffer.str();

    assert(currentFileContent_.size() >= sequenceLength_ &&
           "Loaded file is too small for the sequence length.");

    resetDistribution();
    currentSampleCount_ = 0;
}

// Explicitly loads a new random file when needed
void TextCrawler::loadNextFile() {
    std::uniform_int_distribution<size_t> fileDist(0, filepaths_.size() - 1);
    size_t randomFileIndex = fileDist(generator_);
    loadFile(filepaths_[randomFileIndex]);
}

// Explicitly returns random sequence from current file
std::string TextCrawler::getRandomSequence() {
    if (currentSampleCount_ >= samplesPerFile_) {
        loadNextFile(); // explicitly load new file after N samples
    }

    size_t randomIndex = distribution_(generator_);
    currentSampleCount_++;

    return currentFileContent_.substr(randomIndex, sequenceLength_);
}

// Explicitly returns current loaded file size
size_t TextCrawler::currentFileSize() const {
    return currentFileContent_.size();
}

// Explicitly resets random distribution boundaries for sampling
void TextCrawler::resetDistribution() {
    distribution_ = std::uniform_int_distribution<size_t>(
        0, currentFileContent_.size() - sequenceLength_);
}
