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
    return &flattenedTargetBuffer_[batchIndex * sequenceLength_ * tokenizer_->vocabSize()];
}

// explicitly tokenizes new batch of raw text
void TokenizedDataset::loadData(int batchSize) {
    inputData_.resize(batchSize);
    targetData_.resize(batchSize);
    
    for (int i = 0; i < batchSize; ++i) {
        std::string sequence = textCrawler_->getRandomSequence();
        std::vector<int> tokenIds = tokenizer_->tokenize(sequence);
        
        assert(tokenIds.size() == sequenceLength_);
        
        inputData_[i].resize(0);
        targetData_[i].resize(0);
        
        for (int iToken = 0; iToken < tokenIds.size() - 1; iToken++){
            int token = tokenIds[iToken];
            int nextToken = tokenIds[iToken+1];
            
            inputData_[i].push_back(token);    // inputs explicitly all tokens except last
            targetData_[i].push_back(nextToken);   // targets explicitly next token predictions
        }
        
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
    int vocabSize = (int)tokenizer_->vocabSize();
    flattenedInputBuffer_.resize(batchSize_ * (sequenceLength_ - 1));
    flattenedTargetBuffer_.resize(batchSize_ * (sequenceLength_ - 1) * vocabSize);
    std::fill(flattenedTargetBuffer_.begin(), flattenedTargetBuffer_.end(), 0.0f);

    for (int i = 0; i < batchSize_; ++i) {
        // copy inputs explicitly
        std::copy(inputData_[i].begin(), inputData_[i].end(),
                  flattenedInputBuffer_.begin() + i * (sequenceLength_ - 1));

        // explicitly one-hot encode targets
        for (int j = 0; j < sequenceLength_ - 1; ++j) {
            int tokenID = targetData_[i][j];
            oneHotEncode(flattenedTargetBuffer_, (i * (sequenceLength_ - 1) + j), vocabSize, tokenID);
        }
    }
}

void TokenizedDataset::loadNextBatch(int currentBatchSize) {
    assert(currentBatchSize <= batchSize_);
    loadData(currentBatchSize);
}

float TokenizedDataset::calculateLoss(const float* predictions, int outputDim, const float* targets) {
    float loss = 0.0f;
    //int totalElements = batchSize_ * (sequenceLength_ - 1);

    //for (int i = 0; i < totalElements; ++i) {
        for (int j = 0; j < outputDim; ++j) {
            int index =  j;
            float pred = predictions[index];
            float target = targets[index];
            loss += -(target * logf(pred + 1e-9f)); // explicit CrossEntropy loss calculation
        }
    //}

    return loss;
}

int TokenizedDataset::inputDim() const {
    // input dimension equals sequence length minus 1 (since the last token is used as target)
    return sequenceLength_ - 1;
}

int TokenizedDataset::outputDim() const {
    // output dimension is the vocabulary size (number of possible token IDs)
    return (int)tokenizer_->vocabSize();
}

void TokenizedDataset::oneHotEncode(std::vector<float>& buffer, int index, int vocabSize, int tokenID) {
    int offset = index * vocabSize;
    buffer[offset + tokenID] = 1.0f;
}
