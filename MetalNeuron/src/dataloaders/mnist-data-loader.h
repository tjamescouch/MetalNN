//
//  mnist-data-loader.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#include "dataset.h"

class MNISTDataLoader : public Dataset {
public:
    MNISTDataLoader(const std::string& imagesFilename = "train-images-idx3-ubyte",
                    const std::string& labelsFilename = "train-labels-idx1-ubyte");

    int numSamples() const override;
    int inputDim() const override;
    int outputDim() const override;

    const std::vector<float>& inputAt(int index) override;
    const std::vector<float>& targetAt(int index) override;

private:
    void loadImages(const std::string& imagesPath);
    void loadLabels(const std::string& labelsPath);

    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> targets_;
};
