//
//  view-delegate.h
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//
#ifndef VIEW_DELEGATE_H
#define VIEW_DELEGATE_H

#pragma region Declarations {
#include "computer.h"

class ViewDelegate : public MTK::ViewDelegate
{
    public:
    ViewDelegate( MTL::Device* pDevice );
        virtual ~ViewDelegate() override;
        virtual void drawInMTKView( MTK::View* pView ) override;
        Computer* getComputer() { return _pComputer; }

    private:
        Computer* _pComputer;
};

#pragma endregion Declarations }

#endif
