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
, _pComputer( new Computer( pDevice ) )
{
}

ViewDelegate::~ViewDelegate()
{
    delete _pComputer;
}

void ViewDelegate::drawInMTKView( MTK::View* pView )
{
    pView->setDepthStencilPixelFormat(MTL::PixelFormatDepth32Float);
    pView->setClearDepth(1.0);
}

#pragma endregion ViewDelegate }

