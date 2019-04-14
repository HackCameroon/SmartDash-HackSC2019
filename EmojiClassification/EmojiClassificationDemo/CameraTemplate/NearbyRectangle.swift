//
//  FaceRectangle.swift
//  EmojiClassificationDemo
//
//  Created by Ritesh Pakala on 4/10/19.
//  Copyright © 2019 Ritesh Pakala. All rights reserved.
//

import Foundation
import UIKit

class NearbyRectangle : UIView {
    let nearbyLabel : UILabel = UILabel()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        style()
        setupComponents()
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func style(){
        self.backgroundColor = UIColor.gray
        self.alpha = 0.75
    }
    
    func setupComponents(){
        nearbyLabel.translatesAutoresizingMaskIntoConstraints = false
        nearbyLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        nearbyLabel.textAlignment = .center
        self.addSubview(nearbyLabel)
        
        
        nearbyLabel.topAnchor.constraint(equalTo: self.topAnchor).isActive = true
        nearbyLabel.bottomAnchor.constraint(equalTo: self.bottomAnchor).isActive = true
        nearbyLabel.leftAnchor.constraint(equalTo: self.leftAnchor).isActive = true
        nearbyLabel.rightAnchor.constraint(equalTo: self.rightAnchor).isActive = true
    }
    
    func updateLabel(with data: [String : Double]){
//        let CheckValue = Double(data["no_accident"]!)
//        if (CheckValue <= 5) {
//            nearbyLabel.text = "ACCIDENT FOUND: \(CheckValue)"
//        } else {
//            nearbyLabel.text = "OK: \(CheckValue)"
//        }
        
        nearbyLabel.text = "Nearby Crashes"
        
    }
    
}
