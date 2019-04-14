//
//  AccidentDetection.swift
//  AccidentClassificationDemo
//
//  Created by Ritesh Pakala on 4/10/19.
//  Copyright © 2019 Ritesh Pakala. All rights reserved.
//

import Foundation
import PolarrKit

@available(OSX 10.13.2, *)
public func AccidentDetection() -> SequenceHandler<AccidentDetectionOutput.Output> {
    return ImageToTexture()
        --> ScaleTexture(width: 500, height: 281)
        --> CoreML(url: AccidentClassification.urlOfModelInThisBundle)
        --> AccidentDetectionOutput()
}

@available(OSX 10.13.2, *)
public func AccidentDetection(on image : Image) -> AccidentDetectionOutput.Output? {
    
    
    guard let result = try? AccidentDetection().run(on: image) else {
        return nil
    }
    
    return result
}
