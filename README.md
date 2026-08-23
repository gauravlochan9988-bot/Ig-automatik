# IG-AUTOMATIK

> Ein privater Medien-Workflow für hochwertige Instagram-Fotos und -Videos — vom iPhone direkt zur fertigen Ausgabe.

IG-AUTOMATIK nimmt neue Medien automatisch entgegen, analysiert und bearbeitet sie und erzeugt veröffentlichungsfertige Varianten für Instagram. Die Originale bleiben im Archiv erhalten.

## Was das Projekt kann

| Bereich | Funktion |
| --- | --- |
| **Automatik** | Neue Dateien im Eingangsordner werden automatisch erkannt und verarbeitet |
| **Bildbearbeitung** | Variante A: Natural · Variante B: Cinematic |
| **Instagram-Formate** | Posts im Format 4:5 und Stories im Format 9:16 |
| **Video** | Unterstützung für MP4, MOV und weitere Videoformate; Reels sind vorbereitet |
| **Originale** | Dateien werden sicher archiviert und nicht überschrieben |
| **Mobile** | Private iPhone-Web-App über WLAN oder Tailscale |
| **Qualität** | Master-Dateien, Manifeste und automatische Exportprüfung |

## Der Workflow

```text
                 ┌─────────────────────┐
                 │  iPhone / Eingang   │
                 └──────────┬──────────┘
                            │ Upload
                            ▼
                 ┌─────────────────────┐
                 │     1_EINGANG       │
                 └──────────┬──────────┘
                            │ Watchdog
                            ▼
                 ┌─────────────────────┐
                 │  Grading & QA       │
                 │  Natural / Cinematic│
                 └───────┬───────┬─────┘
                         │       │
                         ▼       ▼
                 ┌──────────┐ ┌──────────┐
                 │ 2_FERTIG │ │ 3_ARCHIV │
                 │ Instagram│ │ Originale │
                 └──────────┘ └──────────┘
```

## Schnellstart

### 1. Watchdog starten

Auf dem Windows-PC in PowerShell:

```powershell
Set-Location 'S:\all my projects\IG-AUTOMATIK\_SYSTEM\app'
& '..\.venv-win\Scripts\python.exe' .\watch.py
```

Danach Dateien in `1_EINGANG/` kopieren. Die fertigen Ergebnisse erscheinen automatisch in `2_FERTIG/`.

### 2. Mobile-App starten

Der Watchdog muss weiterlaufen. In einem zweiten PowerShell-Fenster:

```powershell
Set-Location 'S:\all my projects\IG-AUTOMATIK\_SYSTEM\mobile'
& '..\.venv-win\Scripts\python.exe' .\server.py --project-root 'S:\all my projects\IG-AUTOMATIK'
```

Danach auf dem iPhone öffnen:

```text
http://<IP-des-Windows-PCs>:8787
```

In Safari kann die Seite über **Teilen → Zum Home-Bildschirm** wie eine App installiert werden. Alternativ funktioniert der Zugriff über die Tailscale-IP.

## Projektstruktur

```text
IG-AUTOMATIK/
├── 1_EINGANG/             Neue Fotos und Videos
├── 2_FERTIG/              Fertige Instagram-Ausgaben
│   ├── POSTS/             4:5
│   ├── STORIES/           9:16
│   └── REELS/             9:16, wenn aktiviert
├── 3_ARCHIV/              Originale und Master-Dateien
└── _SYSTEM/
    ├── app/               Pipeline, Watchdog und Tests
    ├── mobile/            iPhone-Web-App und lokaler Server
    ├── config/            Konfiguration und Stilprofil
    ├── manifests/         Export-Informationen
    └── logs/              Laufzeitprotokolle
```

## Mobile-App

Die Mobile-App ist eine schlanke lokale Web-App für den privaten Gebrauch. Sie kann:

- mehrere Fotos oder Videos vom iPhone hochladen;
- den Verarbeitungsstatus anzeigen;
- Natural- und Cinematic-Varianten ansehen;
- fertige Dateien teilen oder in „Fotos“ sichern;
- die Historie durchsuchen und fertige Jobs erneut verarbeiten.

Der Server lauscht standardmäßig auf Port `8787`. Bitte den Port nur im privaten WLAN oder im eigenen Tailscale-Netz verwenden und niemals öffentlich weiterleiten.

Mehr Details: [`_SYSTEM/mobile/README.md`](_SYSTEM/mobile/README.md)
Der native Capacitor-Wrapper für „In Fotos speichern“ und das iOS-Share-Sheet liegt unter [`_SYSTEM/mobile/capacitor/`](_SYSTEM/mobile/capacitor/).

## Konfiguration

Die zentrale Konfiguration liegt lokal in `_SYSTEM/config/config.json` und wird nicht in GitHub gespeichert. Dort werden unter anderem Ausgabeformate, Bildbreiten, Qualität und Videooptionen festgelegt.

Das persönliche Stilprofil liegt in [`_SYSTEM/config/account_style_profile.json`](_SYSTEM/config/account_style_profile.json).

Optionale Vision-Analyse wird über eine lokale `.env` aktiviert:

```text
OPENROUTER_API_KEY=dein-schlüssel
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
```

Die `.env`-Datei gehört niemals in GitHub.

## Tests

Die Pipeline-Tests ausführen:

```powershell
Set-Location 'S:\all my projects\IG-AUTOMATIK\_SYSTEM\app'
& '..\.venv-win\Scripts\python.exe' -m unittest discover -s tests -q
```

## Dokumentation

- [`_SYSTEM/app/README.md`](_SYSTEM/app/README.md) — Pipeline und Bildbearbeitung
- [`_SYSTEM/app/QUICK_START.md`](_SYSTEM/app/QUICK_START.md) — kurzer Einstieg
- [`_SYSTEM/mobile/README.md`](_SYSTEM/mobile/README.md) — iPhone-App und Netzwerkzugriff
- [`_SYSTEM/mobile/capacitor/README.md`](_SYSTEM/mobile/capacitor/README.md) — nativer iOS-Wrapper

## Status

Die aktuelle Version ist für den privaten Betrieb auf einem Windows-PC mit angeschlossenem Projektlaufwerk ausgelegt. Die Mobile-Oberfläche wird ohne Anmeldung betrieben und ist deshalb bewusst nur für ein vertrauenswürdiges internes Netzwerk gedacht.

---

**IG-AUTOMATIK** · Private media workflow · Natural results, cinematic options.
