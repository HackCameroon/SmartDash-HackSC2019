//
//  CameraViewController.swift
//  CameraTemplate
//
//  Created by Ritesh Pakala on 4/08/2019.
//  Copyright © 2019 Polarr, Inc. All rights reserved.
//

import UIKit
import AVFoundation
import MetalKit
import MetalPerformanceShaders
import CoreMotion
import Photos
import Vision

open class CameraViewController : UIViewController {
    
    ///Camera native FPS.
    public var fps : Double = 30
    
    fileprivate var renderingView : PORenderingView!
    fileprivate var logoView : UIImageView!
    
    fileprivate var device : MTLDevice!
    fileprivate var commandQueue : MTLCommandQueue!
    fileprivate var scaleKernel : MPSImageBilinearScale!
    fileprivate var texture : MTLTexture?
    
    fileprivate var captureQueue = DispatchQueue(label: "co.polarr.demo.camera.queue")
    fileprivate var captureSession = AVCaptureSession()
    fileprivate var previewOutput = AVCaptureVideoDataOutput()
    
    fileprivate var textureCache: CVMetalTextureCache?
    
    fileprivate var camera : AVCaptureDevice!
    
    fileprivate var discardableQueue = OperationQueue()
    
    fileprivate var label = UILabel()
    
    public var imageView = UIImageView()
    
    open override var prefersStatusBarHidden: Bool {
        return true
    }
    
    ///Resolution of the video feed from the camera input.
    public var currentResolution : CGSize {
        let formatDescription = camera.activeFormat.formatDescription
        let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
        let resolution = CGSize(width: CGFloat(dimensions.height), height: CGFloat(dimensions.width))
        return resolution
    }
    
    ///Rect of the output fit on screen.
    public var contentRect : CGRect {
        return renderingView.displayRect
    }
    
    open override func viewDidLoad() {
        super.viewDidLoad()
        
        self.view.backgroundColor = UIColor.black
        
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()
        scaleKernel = MPSImageBilinearScale(device: device)
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        
        discardableQueue.maxConcurrentOperationCount = 1
        
        setupViews()
        setupCamera()
    }
    
    open override func viewDidLayoutSubviews() {
        renderingView.contentSize = self.currentResolution
    }
    
    open override func viewWillAppear(_ animated: Bool) {
        captureSession.startRunning()
        
        var finalFormat : AVCaptureDevice.Format?
        var maxFps: Double = 0
        for vFormat in camera.formats {
            let ranges = vFormat.videoSupportedFrameRateRanges
            let description = vFormat.formatDescription
            let dimensions = CMVideoFormatDescriptionGetDimensions(description)
            
            let frameRates = ranges[0]
            if frameRates.maxFrameRate >= maxFps && frameRates.maxFrameRate <= self.fps && dimensions.width < 2000 {
                maxFps = frameRates.maxFrameRate
                finalFormat = vFormat
            }
        }
        
        if maxFps != 0 {
            print("[PolarrCameraViewController] FPS set to \(maxFps).")
            let timeValue = Int64(1200.0 / maxFps)
            let timeScale: Int32 = 1200
            try? camera.lockForConfiguration()
            camera.activeFormat = finalFormat!
            camera.activeVideoMinFrameDuration = CMTimeMake(value: timeValue, timescale: timeScale)
            camera.activeVideoMaxFrameDuration = CMTimeMake(value: timeValue, timescale: timeScale)
            camera.unlockForConfiguration()
        }
        
        
    }
    
    open override func viewWillDisappear(_ animated: Bool) {
        captureSession.stopRunning()
    }
    
    func getRenderFrame() -> CGRect{
        return self.renderingView.mtkView.frame
    }
    
    func updateLabel(withEmotion type : String) {
        label.text = type
    }
    
}

extension CameraViewController {
    
    fileprivate func setupViews() {
        renderingView = PORenderingView()
        renderingView.mtkView.delegate = self
        renderingView.mtkView.device = device
        renderingView.mtkView.framebufferOnly = false
        renderingView.mtkView.enableSetNeedsDisplay = true
        renderingView.mtkView.isPaused = true
        renderingView.backgroundColor = UIColor.black
        renderingView.translatesAutoresizingMaskIntoConstraints = false
        self.view.addSubview(renderingView)
        
        logoView = UIImageView()
        logoView.alpha = 0.5
        logoView.image = UIImage(contentsOfFile: Bundle(for: CameraViewController.self).path(forResource: "Logo", ofType: "png")!)
        logoView.translatesAutoresizingMaskIntoConstraints = false
        logoView.contentMode = .scaleAspectFit
        self.view.addSubview(logoView)
        
        renderingView.leftAnchor.constraint(equalTo: self.view.leftAnchor).isActive = true
        renderingView.rightAnchor.constraint(equalTo: self.view.rightAnchor).isActive = true
        renderingView.topAnchor.constraint(equalTo: self.view.topAnchor).isActive = true
        renderingView.bottomAnchor.constraint(equalTo: self.view.bottomAnchor).isActive = true
        
        logoView.widthAnchor.constraint(equalToConstant: 50).isActive = true
        logoView.heightAnchor.constraint(equalToConstant: 50).isActive = true
        logoView.rightAnchor.constraint(equalTo: self.view.rightAnchor, constant: -25.0).isActive = true
        logoView.topAnchor.constraint(equalTo: self.view.topAnchor, constant: 25.0).isActive = true
        
        label.translatesAutoresizingMaskIntoConstraints = false
        self.view.addSubview(label)
        
        let marginGuide = self.view.safeAreaLayoutGuide
        imageView.translatesAutoresizingMaskIntoConstraints = false
        self.view.addSubview(imageView)
        imageView.leftAnchor.constraint(equalTo: self.view.leftAnchor).isActive = true
        imageView.rightAnchor.constraint(equalTo: self.view.rightAnchor).isActive = true
        imageView.bottomAnchor.constraint(equalTo: marginGuide.bottomAnchor).isActive = true
        imageView.widthAnchor.constraint(equalToConstant: 200).isActive = true
        imageView.heightAnchor.constraint(equalToConstant: 200).isActive = true
        imageView.contentMode = .scaleAspectFit
        
        let margins = self.view.safeAreaLayoutGuide
        
        label.leftAnchor.constraint(equalTo: self.view.leftAnchor, constant: 10).isActive = true
        label.topAnchor.constraint(equalTo: margins.topAnchor).isActive = true
        label.rightAnchor.constraint(equalTo: logoView.leftAnchor).isActive = true
        label.centerYAnchor.constraint(equalTo: logoView.centerYAnchor).isActive = true
        label.font = UIFont(name: ".SFUIText-Bold", size: 16.0)!
        label.textColor = UIColor.white
    }
    
    fileprivate func setupCamera() {
        camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: AVMediaType.video, position: .back)
        
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
        
        do {
            captureSession.beginConfiguration()
            
            let input = try AVCaptureDeviceInput(device: camera)
            
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            
            previewOutput = AVCaptureVideoDataOutput()
            previewOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String : kCVPixelFormatType_32BGRA
            ]
            
            previewOutput.setSampleBufferDelegate(self, queue: captureQueue)
            captureSession.addOutput(previewOutput)
            captureSession.commitConfiguration()
            
            previewOutput.connection(with: .video)!.videoOrientation = .portrait
            previewOutput.connection(with: .video)!.automaticallyAdjustsVideoMirroring = false
            previewOutput.connection(with: .video)!.isVideoMirrored = false
            previewOutput.connection(with: .video)!.preferredVideoStabilizationMode = .off
        }
        catch {
            print("Cannot initialise camera preview.")
        }
        
    }
    
}

extension CameraViewController : AVCaptureVideoDataOutputSampleBufferDelegate {
    
    @objc open func processTexture(_ texture : MTLTexture, _ sampleBuffer: CMSampleBuffer, completion: @escaping (MTLTexture) -> Void) {
        completion(texture)
    }
    
    @objc public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let texture = convertToMTLTexture(sampleBuffer: sampleBuffer) else {
            return
        }
        
        let operation = BlockOperation()
        operation.addExecutionBlock { [weak operation, weak self] in
            guard let theOperation = operation, theOperation.isCancelled == false else {
                return
            }
            
            self?.processTexture(texture, sampleBuffer) { outputTexture in
                self?.texture = outputTexture
                self?.renderingView.mtkView.draw()
            }
        }
        
        discardableQueue.cancelAllOperations()
        discardableQueue.addOperation(operation)
    }
    
    @objc public func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        print("Dropped frame")
    }
    
    fileprivate func convertToMTLTexture(sampleBuffer: CMSampleBuffer?) -> MTLTexture? {
        if let textureCache = textureCache,
            let sampleBuffer = sampleBuffer,
            let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            
            var texture: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
                                                      imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)
            
            if let texture = texture {
                return CVMetalTextureGetTexture(texture)
            }
        }
        return nil
    }
    
}

extension CameraViewController : MTKViewDelegate {
    
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        
    }
    
    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
            let commandBuffer = commandQueue.makeCommandBuffer() else {
                return
        }
        
        guard let texture = texture else {
            return
        }
        
        let scaleX = Double(drawable.texture.width) / Double(texture.width)
        let scaleY = Double(drawable.texture.height) / Double(texture.height)
        
        var transform = MPSScaleTransform(scaleX: scaleX, scaleY: scaleY, translateX: 0, translateY: 0)
        
        withUnsafePointer(to: &transform) { ptr in
            scaleKernel.scaleTransform = ptr
        }
        
        scaleKernel.encode(commandBuffer: commandBuffer,
                           sourceTexture: texture,
                           destinationTexture: drawable.texture)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
    }
    
}
