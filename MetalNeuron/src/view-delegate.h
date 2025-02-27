//
//  view-delegate.h
//  LearnMetalCPP
//n
//  Created by James Couch on 2024-12-07.
//
#ifndef VIEW_DELEGATE_H
#define VIEW_DELEGATE_H

#pragma region Declarations {
#include "neural-engine.h"

class ViewDelegate : public MTK::ViewDelegate
{
    public:
    ViewDelegate( MTL::Device* pDevice );
        virtual ~ViewDelegate() override;
        virtual void drawInMTKView( MTK::View* pView ) override;
        NeuralEngine* getComputer() { return _pComputer; }

    private:
        NeuralEngine* _pComputer;
};

#pragma endregion Declarations }

#endif
