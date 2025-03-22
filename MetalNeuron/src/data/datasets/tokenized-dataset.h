#pragma once

#include "dataset.h"
#include "text-crawler.h"
#include "tokenizer.h"
#include <vector>

class TokenizedDataset : public Dataset {
public:
    TokenizedDataset(TextCrawler* textCrawler, Tokenizer* tokenizer,
                     int sequenceLength, int batchSize);
    virtual ~TokenizedDataset();

    int getDatasetSize() const override;
    const float* getInputDataAt(int batchIndex) const override;
    const float* getTargetDataAt(int batchIndex) const override;

    void loadData(int batchSize) override;
    int numSamples() const override;
    

private:
    void shuffleIndices();
    void loadNextBatch();
    float calculateLoss(const float* predictions, const float* targets, int batchSize) const;
    
    void preprocessBatch(); // explicitly tokenize raw sequences into numeric batches

    TextCrawler* textCrawler_;        // explicitly raw text provider
    Tokenizer* tokenizer_;            // explicitly tokenizer instance

    std::vector<std::vector<float>> inputData_;   // explicitly numeric inputs (token IDs as floats)
    std::vector<std::vector<float>> targetData_;  // explicitly numeric targets (next-token predictions)

    int sequenceLength_;
    int batchSize_;
    std::vector<int> shuffledIndices_;

    std::vector<float> flattenedInputBuffer_;  // explicitly flattened buffers for neural net
    std::vector<float> flattenedTargetBuffer_;

    int currentBatchIndex_; // explicitly tracks current batch position
};
