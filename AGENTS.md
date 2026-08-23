# IG-AUTOMATIK – Arbeitsanweisung für Codex

## Projektziel

IG-AUTOMATIK ist eine private Medienpipeline: Upload vom iPhone, automatische
Verarbeitung auf dem Windows-PC und fertige Instagram-Ausgaben in `2_FERTIG`.
Die Originale bleiben in `3_ARCHIV` erhalten.

## Wichtiger aktueller Arbeitsstand

- `_SYSTEM/app/` enthält Pipeline und Watchdog.
- `_SYSTEM/mobile/` enthält den lokalen Python-Server und die mobile Web-App.
- `_SYSTEM/mobile/capacitor/` enthält den Capacitor-iOS-Wrapper.
- Die native iOS-Schicht ergänzt nur zwei Funktionen: `saveToPhotos` und `shareFile`.
- Die fertige Serverdatei muss unverändert und ohne Resize/Re-Encoding in Fotos
  gespeichert oder über das native Share-Sheet geteilt werden.
- Der iOS-Build und der Gerätetest benötigen Mac + Xcode.

## Arbeitsregeln

- Bestehende Pipeline und Ausgabequalität nicht unnötig verändern.
- Keine Secrets, `.env`, lokalen Konfigurationen oder Nutzerdaten committen.
- Laufzeitdaten unter `_SYSTEM/mobile/data/` sowie `mobile-server*.log` nicht
  automatisch committen.
- Nach Änderungen an `_SYSTEM/mobile/web/` die Dateien mit `npx cap sync ios`
  in das iOS-Projekt kopieren.
- Bei nativen Änderungen sowohl Bild als auch Video prüfen.
- Browser/PWA-Fallback erhalten, native Aktionen aber nicht über einen generischen
  Browser-Download ersetzen.

## Relevante Prüfungen

```bash
node --check _SYSTEM/mobile/web/app.js
cd _SYSTEM/mobile/capacitor
npm install
npx cap sync ios
```

Für den nativen Test muss `IG_AUTOMATIK_SERVER_URL` auf die vom iPhone erreichbare
LAN- oder Tailscale-Adresse des laufenden Mobile-Servers zeigen.
