//
//  FaceRectangle.swift
//  EmojiClassificationDemo
//
//  Created by Ritesh Pakala on 4/10/19.
//  Copyright © 2019 Ritesh Pakala. All rights reserved.
//

import Foundation
import UIKit

class NotificationRectangle : UIView {
    let notificationLabel : UILabel = UILabel()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        style()
        setupComponents()
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func style(){
        self.backgroundColor = UIColor.white
        self.alpha = 0.75
    }
    
    func setupComponents(){
        notificationLabel.translatesAutoresizingMaskIntoConstraints = false
        notificationLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        notificationLabel.textAlignment = .center
        self.addSubview(notificationLabel)
        
        
        notificationLabel.topAnchor.constraint(equalTo: self.topAnchor).isActive = true
        notificationLabel.bottomAnchor.constraint(equalTo: self.bottomAnchor).isActive = true
        notificationLabel.leftAnchor.constraint(equalTo: self.leftAnchor).isActive = true
        notificationLabel.rightAnchor.constraint(equalTo: self.rightAnchor).isActive = true
        
    }
    
    func updateLabel(with data: [String : Double]){
        let CheckValue = Double(data["no_accident"]!)
        if (CheckValue <= 5) {
            notificationLabel.text = "ACCIDENT FOUND: \(CheckValue)"
            self.backgroundColor = UIColor.red
        } else {
            notificationLabel.text = "OK: \(CheckValue)"
            self.backgroundColor = UIColor.white
        }
        
            //notificationLabel.text = "\(data["accident"]) - \(data["no_accident"])"
        
    }

}
