//
//  view-delegate.h
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//
#ifndef VIEW_DELEGATE_H
#define VIEW_DELEGATE_H

#pragma region Declarations {
#include "renderer.h"

class ViewDelegate : public MTK::ViewDelegate
{
    public:
    ViewDelegate( MTL::Device* pDevice );
        virtual ~ViewDelegate() override;
        virtual void drawInMTKView( MTK::View* pView ) override;
        Renderer* getRenederer() { return _pRenderer; }

    private:
        Renderer* _pRenderer;
};

#pragma endregion Declarations }

#endif
