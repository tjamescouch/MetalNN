#include "tokenized-dataset.h"
#include <algorithm>
#include <cassert>
#include <numeric>

// Constructor explicitly initializes members and loads initial batch
TokenizedDataset::TokenizedDataset(TextCrawler* textCrawler, Tokenizer* tokenizer,
                                   int sequenceLength, int batchSize)
: textCrawler_(textCrawler), tokenizer_(tokenizer),
sequenceLength_(sequenceLength), batchSize_(batchSize),
currentBatchIndex_(0) {
    loadData(batchSize_);
}

TokenizedDataset::~TokenizedDataset() {}

// explicitly number of total samples (here infinite, but return large fixed value)
int TokenizedDataset::numSamples() const {
    return 1000000; // large explicit number for continuous training
}

int TokenizedDataset::getDatasetSize() const {
    return batchSize_;
}

// explicitly get flattened input data for batch
const float* TokenizedDataset::getInputDataAt(int batchIndex) const {
    assert(batchIndex < batchSize_);
    return &flattenedInputBuffer_[batchIndex * sequenceLength_];
}

// explicitly get flattened target data for batch
const float* TokenizedDataset::getTargetDataAt(int batchIndex) const {
    assert(batchIndex < batchSize_);
    return &flattenedTargetBuffer_[batchIndex * sequenceLength_];
}

// explicitly tokenizes new batch of raw text
void TokenizedDataset::loadData(int batchSize) {
    inputData_.resize(batchSize);
    targetData_.resize(batchSize);
    
    for (int i = 0; i < batchSize; ++i) {
        std::string sequence = textCrawler_->getRandomSequence();
        auto tokenIds = tokenizer_->tokenize(sequence);
        
        assert(tokenIds.size() == sequenceLength_);
        
        inputData_[i].assign(tokenIds.begin(), tokenIds.end() - 1);    // inputs explicitly all tokens except last
        targetData_[i].assign(tokenIds.begin() + 1, tokenIds.end());   // targets explicitly next token predictions
    }
    
    preprocessBatch();
}

// explicitly shuffles batch indices (for completeness, though batches are random already)
void TokenizedDataset::shuffleIndices() {
    shuffledIndices_.resize(batchSize_);
    std::iota(shuffledIndices_.begin(), shuffledIndices_.end(), 0);
    std::shuffle(shuffledIndices_.begin(), shuffledIndices_.end(), std::mt19937(std::random_device{}()));
}

// explicitly prepares flattened buffers for neural network consumption
void TokenizedDataset::preprocessBatch() {
    flattenedInputBuffer_.resize(batchSize_ * (sequenceLength_ - 1));
    flattenedTargetBuffer_.resize(batchSize_ * (sequenceLength_ - 1));
    
    for (int i = 0; i < batchSize_; ++i) {
        std::copy(inputData_[i].begin(), inputData_[i].end(),
                  flattenedInputBuffer_.begin() + i * (sequenceLength_ - 1));
        std::copy(targetData_[i].begin(), targetData_[i].end(),
                  flattenedTargetBuffer_.begin() + i * (sequenceLength_ - 1));
    }
}

void TokenizedDataset::loadNextBatch() {
    loadData(batchSize_);
}

float TokenizedDataset::calculateLoss(const float* predictions, const float* targets, int batchSize) const {
    // explicitly implement CrossEntropy loss or similar
    float loss = 0.0f;
    int totalElements = batchSize * (sequenceLength_ - 1);
    
    for (int i = 0; i < totalElements; ++i) {
        float pred = predictions[i];
        float target = targets[i];
        loss += -(target * logf(pred + 1e-9f)); // simplified example explicitly
    }
    
    return loss / totalElements;
}
