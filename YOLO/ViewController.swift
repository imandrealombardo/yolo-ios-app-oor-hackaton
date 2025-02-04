//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLO11 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision

var mlModel = try! yolov8m(configuration: mlmodelConfig).model
var mlmodelConfig: MLModelConfiguration = {
  let config = MLModelConfiguration()

  if #available(iOS 17.0, *) {
    config.setValue(1, forKey: "experimentalMLE5EngineUsage")
  }

  return config
}()

class ViewController: UIViewController {
  @IBOutlet var videoPreview: UIView!
  @IBOutlet var View0: UIView!
  @IBOutlet var playButtonOutlet: UIBarButtonItem!
  @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
  @IBOutlet var slider: UISlider!
  @IBOutlet var sliderConf: UISlider!
  @IBOutlet weak var sliderConfLandScape: UISlider!
  @IBOutlet var sliderIoU: UISlider!
  @IBOutlet weak var sliderIoULandScape: UISlider!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelZoom: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var labelSlider: UILabel!
  @IBOutlet weak var labelSliderConf: UILabel!
  @IBOutlet weak var labelSliderConfLandScape: UILabel!
  @IBOutlet weak var labelSliderIoU: UILabel!
  @IBOutlet weak var labelSliderIoULandScape: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var focus: UIImageView!
  @IBOutlet weak var toolBar: UIToolbar!

  let selection = UISelectionFeedbackGenerator()
  var detector = try! VNCoreMLModel(for: mlModel)
  var session: AVCaptureSession!
  var videoCapture: VideoCapture!
  var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  // var cameraOutput: AVCapturePhotoOutput!
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false
  var lastPixelBufferForSaving: CVPixelBuffer?

  // Developer mode
  let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")  // developer mode selected in settings
  let save_detections = false  // write every detection to detections.txt
  let save_frames = false  // write every frame to frames.txt
    

  lazy var visionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: {
        [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    return request
  }()

  override func viewDidLoad() {
    super.viewDidLoad()
    setUpBoundingBoxViews()
    setUpOrientationChangeNotification()
    startVideo()
    // setModel()
  }

  override func viewWillTransition(
    to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
  ) {
    super.viewWillTransition(to: size, with: coordinator)

    if size.width > size.height {
      labelSliderConf.isHidden = true
      sliderConf.isHidden = true
      labelSliderIoU.isHidden = true
      sliderIoU.isHidden = true
      toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = false
      sliderConfLandScape.isHidden = false
      labelSliderIoULandScape.isHidden = false
      sliderIoULandScape.isHidden = false

    } else {
      labelSliderConf.isHidden = false
      sliderConf.isHidden = false
      labelSliderIoU.isHidden = false
      sliderIoU.isHidden = false
      toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(nil, forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = true
      sliderConfLandScape.isHidden = true
      labelSliderIoULandScape.isHidden = true
      sliderIoULandScape.isHidden = true
    }
    self.videoCapture.previewLayer?.frame = CGRect(
      x: 0, y: 0, width: size.width, height: size.height)

  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation()
    //      frameSizeCaptured = false
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  func setModel() {
      
    mlModel = try! yolov8m(configuration: mlmodelConfig).model

    /// VNCoreMLModel
    detector = try! VNCoreMLModel(for: mlModel)
    detector.featureProvider = ThresholdProvider()

    /// VNCoreMLRequest
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: { [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    visionRequest = request
  }

  /// Update thresholds from slider values
  @IBAction func sliderChanged(_ sender: Any) {
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
  }

  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  var colors: [String: UIColor] = [:]
  let ultralyticsColorsolors: [UIColor] = [
    UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),  // #042AFF
    UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),  // #0BDBEB
    UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),  // #F3F3F3
    UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),  // #00DFB7
    UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),  // #111F68
    UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),  // #FF6FDD
    UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),  // #FF444F
    UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),  // #CCED00
    UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),  // #00F344
    UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),  // #BD00FF
    UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),  // #00B4FF
    UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),  // #DD00BA
    UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),  // #00FFFF
    UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),  // #26C000
    UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),  // #01FFB3
    UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),  // #7D24FF
    UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),  // #7B0068
    UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),  // #FF1B6C
    UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),  // #FC6D2F
    UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),  // #A2FF0B
  ]

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

    // Retrieve class labels directly from the CoreML model's class labels, if available.
    guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
      fatalError("Class labels are missing from the model description")
    }

    // Assign random colors to the classes.
    var count = 0
    for label in classLabels {
      let color = ultralyticsColorsolors[count]
      count += 1
      if count > 19 {
        count = 0
      }
      colors[label] = color

    }
  }

  func startVideo() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    videoCapture.setUp(sessionPreset: .photo) { success in
      // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxViews {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  func predict(sampleBuffer: CMSampleBuffer) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      self.lastPixelBufferForSaving = pixelBuffer
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
      }
      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      let imageOrientation: CGImagePropertyOrientation
      switch UIDevice.current.orientation {
      case .portrait:
        imageOrientation = .up
      case .portraitUpsideDown:
        imageOrientation = .down
      case .landscapeLeft:
        imageOrientation = .up
      case .landscapeRight:
        imageOrientation = .up
      case .unknown:
        imageOrientation = .up
      default:
        imageOrientation = .up
      }

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
        do {
          try handler.perform([visionRequest])
        } catch {
          print(error)
        }
      }

      currentBuffer = nil
    }
  }

  func processObservations(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      if let results = request.results as? [VNRecognizedObjectObservation] {
        self.show(predictions: results)
          
          // --- Step 1: Check if at least one "container" is detected.
          let containerDetected = results.contains { observation in
              if let label = observation.labels.first?.identifier.lowercased() {
                  return label == "container"
              }
              return false
          }
          
          // Only proceed if a container is detected.
          if containerDetected {
              // --- Step 2: Identify sensitive objects and collect their bounding boxes.
              // Define your sensitive classes.
              let sensitiveClasses: Set<String> = ["person", "license plate"]
              var sensitiveBoxes = [CGRect]()
              
              // For each observation that is sensitive, convert its normalized bounding box to image coordinates.
              // (Assume 'image' is created from your saved pixel buffer.)
              if let pixelBuffer = self.lastPixelBufferForSaving,
                 let image = self.imageFromPixelBuffer(pixelBuffer: pixelBuffer) {
                  
                  let imageSize = image.size
                  for observation in results {
                      if let label = observation.labels.first?.identifier.lowercased(),
                         sensitiveClasses.contains(label) {
                          let normRect = observation.boundingBox
                          // VNImageRectForNormalizedRect converts a normalized rect (origin bottom-left)
                          // into pixel coordinates (origin top-left) given the image width and height.
                          let rectInImage = VNImageRectForNormalizedRect(normRect, Int(imageSize.width), Int(imageSize.height))
                          sensitiveBoxes.append(rectInImage)
                      }
                  }
                  
                  // --- Step 3: Blur the sensitive regions.
                  if !sensitiveBoxes.isEmpty, let blurredImage = self.blurSensitiveAreas(in: image, boxes: sensitiveBoxes, blurRadius: 10) {
                      // Save the blurred image (using your custom saveDetetction(_:) method).
                      self.saveDetection(image: blurredImage, predictions: results)
//                      // Optionally clear the pixel buffer so this frame isn’t saved again.
//                      self.lastPixelBufferForSaving = nil
                  }
                else {
                  self.saveDetection(image: image, predictions: results)
                }
                // Optionally clear the pixel buffer so this frame isn’t saved again.
                self.lastPixelBufferForSaving = nil
              }
          }
      } else {
        self.show(predictions: [])
      }
    }
  }

  func show(predictions: [VNRecognizedObjectObservation]) {
    var str = ""
    // date
    let date = Date()
    let calendar = Calendar.current
    let hour = calendar.component(.hour, from: date)
    let minutes = calendar.component(.minute, from: date)
    let seconds = calendar.component(.second, from: date)
    let nanoseconds = calendar.component(.nanosecond, from: date)
    let sec_day =
      Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day

    let width = videoPreview.bounds.width  // 375 pix
    let height = videoPreview.bounds.height  // 812 pix

    if UIDevice.current.orientation == .portrait {

      // ratio = videoPreview AR divided by sessionPreset AR
      var ratio: CGFloat = 1.0
      if videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)  // .photo
      } else {
        ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
      }

      for i in 0..<boundingBoxViews.count {
        if i < predictions.count {
          let prediction = predictions[i]

          var rect = prediction.boundingBox  // normalized xywh, origin lower left
          switch UIDevice.current.orientation {
          case .portraitUpsideDown:
            rect = CGRect(
              x: 1.0 - rect.origin.x - rect.width,
              y: 1.0 - rect.origin.y - rect.height,
              width: rect.width,
              height: rect.height)
          case .landscapeLeft:
            rect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .landscapeRight:
            rect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .unknown:
            print("The device orientation is unknown, the predictions may be affected")
            fallthrough
          default: break
          }

          if ratio >= 1 {  // iPhone ratio = 1.218
            let offset = (1 - ratio) * (0.5 - rect.minX)
            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
            rect = rect.applying(transform)
            rect.size.width *= ratio
          } else {  // iPad ratio = 0.75
            let offset = (ratio - 1) * (0.5 - rect.maxY)
            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
            rect = rect.applying(transform)
            ratio = (height / width) / (3.0 / 4.0)
            rect.size.height /= ratio
          }

          // Scale normalized to pixels [375, 812] [width, height]
          rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

          // The labels array is a list of VNClassificationObservation objects,
          // with the highest scoring class first in the list.
          let bestClass = prediction.labels[0].identifier
          let confidence = prediction.labels[0].confidence
          // print(confidence, rect)  // debug (confidence, xywh) with xywh origin top left (pixels)
          let label = String(format: "%@ %.1f", bestClass, confidence * 100)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          // Show the bounding box.
          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: colors[bestClass] ?? UIColor.white,
            alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
        } else {
          boundingBoxViews[i].hide()
        }
      }
    } else {
      let frameAspectRatio = longSide / shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / shortSide
        scaleX = scaleY
        offsetX = (longSide * scaleX - width) / 2
      } else {
        scaleX = width / longSide
        scaleY = scaleX
        offsetY = (shortSide * scaleY - height) / 2
      }

      for i in 0..<boundingBoxViews.count {
        if i < predictions.count {
          let prediction = predictions[i]

          var rect = prediction.boundingBox

          rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
          rect.origin.y =
            height
            - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY)
          rect.size.width *= longSide * scaleX
          rect.size.height *= shortSide * scaleY

          let bestClass = prediction.labels[0].identifier
          let confidence = prediction.labels[0].confidence

          let label = String(format: "%@ %.1f", bestClass, confidence * 100)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          // Show the bounding box.
          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: colors[bestClass] ?? UIColor.white,
            alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
        } else {
          boundingBoxViews[i].hide()
        }
      }
    }
  }
    
    func imageFromPixelBuffer(pixelBuffer: CVPixelBuffer) -> UIImage? {
      let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
      let context = CIContext()
      if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
        return UIImage(cgImage: cgImage)
      }
      return nil
    }
  
  func saveDetection(image: UIImage, predictions: [VNRecognizedObjectObservation]) {
      // Generate a filename using the current date/time.
      let dateFormatter = DateFormatter()
      dateFormatter.dateFormat = "yyyyMMdd_HHmmssSSS"
      let dateString = dateFormatter.string(from: Date())
      let fileNameBase = "detection_\(dateString)"
      
      // Locate the "Detections" folder in the app’s Documents directory.
      guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
          print("Could not locate Documents folder.")
          return
      }
      let detectionsFolderURL = documentsURL.appendingPathComponent("Detections")
      
      // Ensure the "Detections" folder exists.
      if !FileManager.default.fileExists(atPath: detectionsFolderURL.path) {
          do {
              try FileManager.default.createDirectory(at: detectionsFolderURL, withIntermediateDirectories: true, attributes: nil)
              print("Created Detections folder at: \(detectionsFolderURL.path)")
          } catch {
              print("Error creating folder: \(error.localizedDescription)")
              return
          }
      }
      
      // Save the image as a JPEG.
      let imageURL = detectionsFolderURL.appendingPathComponent(fileNameBase + ".jpg")
      if let imageData = image.jpegData(compressionQuality: 0.5) {
          do {
              try imageData.write(to: imageURL)
              print("Saved image at \(imageURL)")
          } catch {
              print("Error saving image: \(error)")
          }
      }
      
      // Build the metadata for each prediction.
      var predictionsMetadata = [[String: Any]]()
      for prediction in predictions {
          if let bestLabel = prediction.labels.first?.identifier {
              let meta: [String: Any] = [
                  "label": bestLabel,
                  "confidence": prediction.labels.first?.confidence ?? 0,
                  "boundingBox": [
                      "x": prediction.boundingBox.origin.x,
                      "y": prediction.boundingBox.origin.y,
                      "width": prediction.boundingBox.size.width,
                      "height": prediction.boundingBox.size.height
                  ]
              ]
              predictionsMetadata.append(meta)
          }
      }
      
      // Create the metadata dictionary.
      let metadata: [String: Any] = [
          "timestamp": dateString,
          "predictions": predictionsMetadata
          // TO ADD: COORDINATE
      ]
      
      // Save the metadata as a JSON file.
      let metadataURL = detectionsFolderURL.appendingPathComponent(fileNameBase + ".json")
      do {
          let jsonData = try JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted)
          try jsonData.write(to: metadataURL)
          print("Saved metadata at \(metadataURL)")
      } catch {
          print("Error saving metadata: \(error)")
      }
  }
  
  func blurSensitiveAreas(in image: UIImage, boxes: [CGRect], blurRadius: Double = 20) -> UIImage? {
      // Convert the UIImage to a CIImage.
      guard let ciImage = CIImage(image: image) else { return nil }
      var outputImage = ciImage
      let context = CIContext(options: nil)
      
      for box in boxes {
          // Crop the region to blur.
          let cropped = outputImage.cropped(to: box)
          
          // Apply a Gaussian blur filter to the cropped area.
          if let blurFilter = CIFilter(name: "CIGaussianBlur") {
              blurFilter.setValue(cropped, forKey: kCIInputImageKey)
              blurFilter.setValue(blurRadius, forKey: kCIInputRadiusKey)
              guard let blurredCropped = blurFilter.outputImage else { continue }
              // The blur filter may expand the image extent; crop back to the original box.
              let blurredRegion = blurredCropped.cropped(to: box)
              
              // Composite the blurred region over the current output image.
              if let compositeFilter = CIFilter(name: "CISourceOverCompositing") {
                  compositeFilter.setValue(blurredRegion, forKey: kCIInputImageKey)
                  compositeFilter.setValue(outputImage, forKey: kCIInputBackgroundImageKey)
                  if let composited = compositeFilter.outputImage {
                      outputImage = composited
                  }
              }
          }
      }
      
      // Render the final composited image.
      if let cgImage = context.createCGImage(outputImage, from: outputImage.extent) {
          return UIImage(cgImage: cgImage)
      }
      return nil
  }
}  // ViewController class End

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
    predict(sampleBuffer: sampleBuffer)
  }
}
