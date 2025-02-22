//
//  view-delegate.cpp
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//

#include "view-delegate.h"

#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate( MTL::Device* pDevice )
: MTK::ViewDelegate()
, _pRenderer( new Renderer( pDevice ) )
{
}

ViewDelegate::~ViewDelegate()
{
    delete _pRenderer;
}

void ViewDelegate::drawInMTKView( MTK::View* pView )
{
    pView->setDepthStencilPixelFormat(MTL::PixelFormatDepth32Float);
    pView->setClearDepth(1.0);

    _pRenderer->draw( pView );
}

#pragma endregion ViewDelegate }

