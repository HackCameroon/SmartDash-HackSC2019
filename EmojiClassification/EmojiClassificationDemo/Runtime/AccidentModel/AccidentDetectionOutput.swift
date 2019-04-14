//
//  AccidentDetectionOutput.swift
//  AccidentClassificationDemo
//
//  Created by Ritesh Pakala on 4/10/19.
//  Copyright © 2019 Ritesh Pakala. All rights reserved.
//

import Foundation
import PolarrKit

public class AccidentDetectionOutput : Handler, HandlerContextRecipient {
    public typealias Input = [String : Any]
    public typealias Output = [String:Double]
    
    weak var context : HandlerContext?
    
    public init(){
        
    }
    
    public func invalidate(context: HandlerContext) {
        self.context = context
    }
    
    public func handle(input: [String : Any], state: HandlerState) throws -> [String:Double]? {
        
        if let accidentConfidence = input["prob"] as? [String:Double] {
            return accidentConfidence
        }
        
        return nil
    }
    
}
