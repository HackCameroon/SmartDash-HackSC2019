//
//  PORenderingView.swift
//  Powder
//
//  Created by Daniel Vershinin on 14/10/2016.
//  Copyright © 2016 Polarr. All rights reserved.
//

import Foundation
import MetalKit

public typealias POBaseView = UIView
public typealias POEdgeInsets = UIEdgeInsets

public enum PORenderingViewMode {
    case scaleAspectFit
    case scaleAspectFill
}

public class PORenderingView : POBaseView {
    
    var contentExtendedEdges = POEdgeInsets(top: 0, left: 0, bottom: 0, right: 0)
    
    var contentSize : CGSize = CGSize(width: 640, height: 480) {
        
        didSet {
            layoutMetalView()
        }
        
    }
    
    public var scalingMode : PORenderingViewMode = .scaleAspectFit {
        
        didSet {
            layoutMetalView()
        }
        
    }
    
    public fileprivate(set) var displayRect = CGRect.zero
    
    public let mtkView = MTKView()
    
    public required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupMetalView()
    }
    
    override init(frame frameRect: CGRect) {
        super.init(frame: frameRect)
        setupMetalView()
    }
    
    #if os(iOS) || os(watchOS) || os(tvOS)
    open override func layoutSubviews() {
        super.layoutSubviews()
        layoutMetalView()
    }
    #else
    public override func layout() {
        super.layout()
        layoutMetalView()
    }
    #endif
    
    fileprivate func setupMetalView() {
        scalingMode = .scaleAspectFit
        
        #if os(iOS) || os(watchOS) || os(tvOS)
        setBackgroundColor(color: UIColor(white: 0.0, alpha: 1.0).cgColor)
        #else
        setBackgroundColor(color: NSColor(white: 0.0, alpha: 1.0).cgColor)
        #endif
        
        self.addSubview(mtkView)
    }
    
    fileprivate func layoutMetalView() {
        
        if contentSize.width < self.frame.width &&
            contentSize.height < self.frame.height {
            mtkView.frame.size = contentSize
        }
        else {
            let xScale = contentSize.width / self.frame.width
            let yScale = contentSize.height / self.frame.height
            
            let scale = scalingMode == .scaleAspectFit ? max(xScale, yScale) : min(xScale, yScale)
            mtkView.frame.size = CGSize(width: contentSize.width / scale, height: contentSize.height / scale)
        }
        
        mtkView.frame.origin = CGPoint(x: (self.frame.width - mtkView.frame.width) / 2,
                                       y: (self.frame.height - mtkView.frame.height) / 2)
        
        mtkView.frame.origin.x += contentExtendedEdges.left
        mtkView.frame.size.width += contentExtendedEdges.right
        mtkView.frame.origin.y += contentExtendedEdges.top
        mtkView.frame.size.height += contentExtendedEdges.bottom
        
        self.displayRect = mtkView.frame
    }
    
    func setBackgroundColor(color: CGColor) {
        #if os(iOS) || os(watchOS) || os(tvOS)
        self.layer.backgroundColor = color
        #else
        self.layer?.backgroundColor = color
        #endif
    }
    
    
}

