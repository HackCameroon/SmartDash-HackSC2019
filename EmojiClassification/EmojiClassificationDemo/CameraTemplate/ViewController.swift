//
//  ViewController.swift
//  EmojiClassification
//
//  Created by Ritesh Pakala on 3/27/19.
//  Copyright © 2019 Polarr. All rights reserved.
//

import PolarrKit
import UIKit
import AVKit
import Vision


class ViewController: CameraViewController {
    
    fileprivate var rectanglesRequest : VNDetectFaceRectanglesRequest!
    fileprivate var landmarksRequest : VNDetectFaceLandmarksRequest!
    fileprivate var rectanglesHandler : VNSequenceRequestHandler!
    
    var displayFrame : CGRect = CGRect.zero
    var notificationRectangle : NotificationRectangle = NotificationRectangle()
    var nearbyRectangle : NearbyRectangle = NearbyRectangle()
    var accidentDetection = AccidentDetection()
    var accidentFrameLimiter = 30
    let accidentFrameTillNextInference = 30
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //These are requests that we are asking Swift to allow preparation for
        //All of these relate to retrieving Facial data to provide
        //valuable results towards face tracking
        self.rectanglesRequest = VNDetectFaceRectanglesRequest()
        self.landmarksRequest = VNDetectFaceLandmarksRequest()
        self.rectanglesHandler = VNSequenceRequestHandler()
        
        
        notificationRectangle.translatesAutoresizingMaskIntoConstraints = false
        self.view.addSubview(notificationRectangle)
        
        nearbyRectangle.translatesAutoresizingMaskIntoConstraints = false
        self.view.addSubview(nearbyRectangle)
        
        let portWidth = min(UIScreen.main.bounds.width, UIScreen.main.bounds.height)
        
        notificationRectangle.leftAnchor.constraint(equalTo: self.view.leftAnchor).isActive = true
        notificationRectangle.topAnchor.constraint(equalTo: self.view.topAnchor).isActive = true
        notificationRectangle.heightAnchor.constraint(equalToConstant: 100).isActive = true
        notificationRectangle.widthAnchor.constraint(equalToConstant: portWidth).isActive = true
        
//        let portWidth = min(UIScreen.main.bounds.width, UIScreen.main.bounds.height)
        
        nearbyRectangle.leftAnchor.constraint(equalTo: self.view.leftAnchor).isActive = true
        nearbyRectangle.bottomAnchor.constraint(equalTo: self.view.bottomAnchor).isActive = true
        nearbyRectangle.heightAnchor.constraint(equalToConstant: 100).isActive = true
        nearbyRectangle.widthAnchor.constraint(equalToConstant: portWidth).isActive = true
    
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(true)
        
        displayFrame = self.getRenderFrame()
        
    }
    
    override func processTexture(_ texture: MTLTexture, _ sampleBuffer: CMSampleBuffer, completion: @escaping (MTLTexture) -> Void) {
        
    
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer){
            
            let benchmarkTime = CFAbsoluteTimeGetCurrent()
            if self.accidentFrameLimiter >= self.accidentFrameTillNextInference, let accidentConfidence = try? self.accidentDetection.run(on: pixelBuffer) {
                print("[MODEL RUNTIME] \(CFAbsoluteTimeGetCurrent() - benchmarkTime)s\n\n")
                self.accidentFrameLimiter = 0
                DispatchQueue.main.async {
                    self.notificationRectangle.updateLabel(with: accidentConfidence)
                    print("\(accidentConfidence)\n\n\n")
                }
            } else {
//                self.accidentDetection.printDebuggingGroups()
                print("[ViewController] Failed to run model.\n\n\n")
            }
            
            self.accidentFrameLimiter += 1
        }else {
            
            print("[ViewController] Failed to convert to CVPixelBuffer.")
        }
        
        completion(texture)
    }
    
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(inputImage, from: inputImage.extent) {
            return cgImage
        }
        return nil
    }
}

