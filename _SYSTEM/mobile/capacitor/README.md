# IG-AUTOMATIK iPhone-App

Dieser Ordner enthält den kleinen Capacitor-Wrapper für die bestehende Mobile-Web-App.
Die Oberfläche und die Server-Pipeline bleiben unverändert. Die native iOS-Schicht ergänzt:

- `saveToPhotos` – fertige Bild- oder Videodatei direkt in die iPhone-Fotomediathek speichern
- `shareFile` – dieselbe Originaldatei über das native iOS-Share-Sheet teilen

## Voraussetzungen

Die iOS-App muss auf einem Mac mit Xcode gebaut und signiert werden. Windows kann die
Web-/Capacitor-Struktur vorbereiten, aber kein iOS-Projekt kompilieren.

## Einrichtung

```bash
npm install
npx cap add ios
npx cap sync ios
npx cap open ios
```

Vor dem ersten echten Test muss beim Synchronisieren die vom iPhone erreichbare
Adresse des IG-AUTOMATIK-Servers gesetzt werden. Das kann eine lokale LAN-Adresse
oder eine HTTPS-Adresse sein. Bei einer lokalen HTTP-Adresse bleibt `cleartext: true`
aktiv:

```bash
IG_AUTOMATIK_SERVER_URL=http://<PC-IP>:8787 npx cap sync ios
```

Die Adresse wird nur in die native Capacitor-Konfiguration übernommen; die Web-App
bleibt weiterhin dieselbe Datei unter `../web`.

Nach Änderungen an `../web`:

```bash
npx cap copy ios
```

## iOS-Berechtigung

Das Projekt benötigt den Schlüssel `NSPhotoLibraryAddUsageDescription` in der App-
`Info.plist`. Der native Bridge-Code fordert ausschließlich die Berechtigung zum
Hinzufügen von Fotos/Videos an; vorhandene Fotos werden nicht gelesen.
