import UIKit
import Capacitor
import Foundation
import Photos

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = scene as? UIWindowScene else { return }

        window = UIWindow(windowScene: windowScene)
        window?.rootViewController = IGAutomatikViewController()
        window?.makeKeyAndVisible()

        SceneDelegateProxy.shared.scene(scene, willConnectTo: session, options: connectionOptions)
    }

    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        SceneDelegateProxy.shared.scene(scene, openURLContexts: URLContexts)
    }

    func scene(_ scene: UIScene, continue userActivity: NSUserActivity) {
        SceneDelegateProxy.shared.scene(scene, continue: userActivity)
    }
}

final class IGAutomatikViewController: CAPBridgeViewController {
    override func capacitorDidLoad() {
        bridge?.registerPluginInstance(IGMediaPlugin())
    }
}

@objc(IGMediaPlugin)
public final class IGMediaPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "IGMediaPlugin"
    public let jsName = "IGMedia"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "saveToPhotos", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "shareFile", returnType: CAPPluginReturnPromise),
    ]

    @objc func saveToPhotos(_ call: CAPPluginCall) {
        guard let request = mediaRequest(from: call) else { return }

        requestPhotoAddPermission { [weak self] granted in
            guard granted else {
                call.reject("Fotos-Berechtigung wurde nicht erteilt.", "photos_permission_denied")
                return
            }
            self?.downloadMedia(request) { result in
                switch result {
                case .failure(let error):
                    call.reject(error.localizedDescription, "download_failed")
                case .success(let fileURL):
                    self?.save(fileURL: fileURL, kind: request.kind, filename: request.filename, call: call)
                }
            }
        }
    }

    @objc func shareFile(_ call: CAPPluginCall) {
        guard let request = mediaRequest(from: call) else { return }

        downloadMedia(request) { [weak self] result in
            switch result {
            case .failure(let error):
                call.reject(error.localizedDescription, "download_failed")
            case .success(let fileURL):
                self?.presentShareSheet(fileURL: fileURL, filename: request.filename, call: call)
            }
        }
    }

    private struct MediaRequest {
        let url: URL
        let filename: String
        let kind: String
    }

    private func mediaRequest(from call: CAPPluginCall) -> MediaRequest? {
        guard let rawURL = call.getString("url"),
              let url = URL(string: rawURL),
              url.scheme == "http" || url.scheme == "https" else {
            call.reject("Die Datei-Adresse ist ungültig.", "invalid_file_url")
            return nil
        }

        let kind = call.getString("kind") == "video" ? "video" : "image"
        let requestedName = call.getString("filename") ?? (kind == "video" ? "IG-AUTOMATIK.mp4" : "IG-AUTOMATIK.jpg")
        let filename = sanitizedFilename(requestedName, kind: kind)
        return MediaRequest(url: url, filename: filename, kind: kind)
    }

    private func sanitizedFilename(_ value: String, kind: String) -> String {
        let base = URL(fileURLWithPath: value).lastPathComponent
        let ext = URL(fileURLWithPath: base).pathExtension.isEmpty
            ? (kind == "video" ? "mp4" : "jpg")
            : URL(fileURLWithPath: base).pathExtension
        let stem = URL(fileURLWithPath: base).deletingPathExtension().lastPathComponent
        let safeStem = stem.replacingOccurrences(of: "[^A-Za-z0-9._-]", with: "_", options: .regularExpression)
        return "\(safeStem.isEmpty ? "IG-AUTOMATIK" : safeStem).\(ext.lowercased())"
    }

    private func requestPhotoAddPermission(completion: @escaping (Bool) -> Void) {
        let accessLevel: PHAccessLevel = .addOnly
        let status = PHPhotoLibrary.authorizationStatus(for: accessLevel)
        if status == .authorized || status == .limited {
            completion(true)
            return
        }
        if status == .denied || status == .restricted {
            completion(false)
            return
        }
        PHPhotoLibrary.requestAuthorization(for: accessLevel) { newStatus in
            DispatchQueue.main.async {
                completion(newStatus == .authorized || newStatus == .limited)
            }
        }
    }

    private func downloadMedia(_ request: MediaRequest, completion: @escaping (Result<URL, Error>) -> Void) {
        var urlRequest = URLRequest(url: request.url)
        urlRequest.cachePolicy = .reloadIgnoringLocalCacheData
        urlRequest.timeoutInterval = 300

        URLSession.shared.downloadTask(with: urlRequest) { temporaryURL, response, error in
            if let error {
                completion(.failure(error))
                return
            }
            if let httpResponse = response as? HTTPURLResponse,
               !(200...299).contains(httpResponse.statusCode) {
                completion(.failure(Self.pluginError("Der Server konnte die Datei nicht liefern.")))
                return
            }
            guard let temporaryURL else {
                completion(.failure(Self.pluginError("Die Datei wurde leer zurückgegeben.")))
                return
            }

            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(URL(fileURLWithPath: request.filename).pathExtension)
            do {
                try FileManager.default.copyItem(at: temporaryURL, to: destination)
                completion(.success(destination))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }

    private func save(fileURL: URL, kind: String, filename: String, call: CAPPluginCall) {
        PHPhotoLibrary.shared().performChanges({
            let creationRequest = PHAssetCreationRequest.forAsset()
            let resourceType: PHAssetResourceType = kind == "video" ? .video : .photo
            let options = PHAssetResourceCreationOptions()
            creationRequest.addResource(with: resourceType, fileURL: fileURL, options: options)
        }) { [weak self] success, error in
            self?.removeTemporaryFile(fileURL)
            DispatchQueue.main.async {
                if success {
                    call.resolve(["saved": true, "filename": filename])
                } else {
                    call.reject(error?.localizedDescription ?? "Die Datei konnte nicht in Fotos gespeichert werden.", "photos_save_failed")
                }
            }
        }
    }

    private func presentShareSheet(fileURL: URL, filename: String, call: CAPPluginCall) {
        DispatchQueue.main.async { [weak self] in
            guard let self, let root = self.bridge?.viewController else {
                self?.removeTemporaryFile(fileURL)
                call.reject("Die iOS-Ansicht ist nicht verfügbar.", "share_unavailable")
                return
            }

            let shareSheet = UIActivityViewController(activityItems: [fileURL], applicationActivities: nil)
            shareSheet.completionWithItemsHandler = { [weak self] _, completed, _, error in
                self?.removeTemporaryFile(fileURL)
                if let error {
                    call.reject(error.localizedDescription, "share_failed")
                } else {
                    call.resolve(["shared": completed, "filename": filename])
                }
            }
            if let popover = shareSheet.popoverPresentationController {
                popover.sourceView = root.view
                popover.sourceRect = CGRect(x: root.view.bounds.midX, y: root.view.bounds.midY, width: 1, height: 1)
            }
            var presenter = root
            while let presented = presenter.presentedViewController {
                presenter = presented
            }
            presenter.present(shareSheet, animated: true)
        }
    }

    private func removeTemporaryFile(_ fileURL: URL) {
        try? FileManager.default.removeItem(at: fileURL)
    }

    private static func pluginError(_ message: String) -> NSError {
        NSError(domain: "IGMediaPlugin", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
