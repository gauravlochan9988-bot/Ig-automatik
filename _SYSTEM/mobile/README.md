# IG-AUTOMATIK Mobile

Das ist eine separate iPhone-Oberfläche. Die bestehende IG-AUTOMATIK-Pipeline
wird nicht importiert, verändert oder ersetzt.

## Start auf dem Windows-PC

Das vorhandene IG-AUTOMATIK-Watchdog-Fenster muss weiterlaufen. Danach in
PowerShell:

```powershell
Set-Location 'S:\all my projects\IG-AUTOMATIK\_SYSTEM\mobile'
& 'S:\all my projects\IG-AUTOMATIK\_SYSTEM\.venv-win\Scripts\python.exe' .\server.py --project-root 'S:\all my projects\IG-AUTOMATIK'
```

Für einen unsichtbaren Start ohne Konsolenfenster kannst du auch
`start-mobile-hidden.vbs` doppelt anklicken.

Die Weboberfläche läuft danach auf Port `8787`.

## Auf dem iPhone öffnen

Im gleichen WLAN im Browser öffnen:

```text
http://<IP-des-Windows-PCs>:8787
```

In Safari: Teilen → **Zum Home-Bildschirm**. Damit ist die Oberfläche wie eine
App auf dem iPhone verfügbar.

## Über Tailscale

1. Tailscale auf PC und iPhone starten.
2. Die Tailscale-IP des PCs ermitteln, zum Beispiel mit `tailscale ip -4`.
3. Auf dem iPhone öffnen: `http://<Tailscale-IP>:8787`.

Es gibt bewusst keine Anmeldung. Deshalb darf der Port nur im privaten WLAN
oder im eigenen Tailscale-Netz erreichbar sein und nicht per Router ins
öffentliche Internet weitergeleitet werden.

## Wichtiger Ablauf

Die App lädt eine Datei zuerst als temporäre Datei hoch und verschiebt sie erst
nach vollständigem Upload atomar nach `1_EINGANG`. Der vorhandene Watchdog
verarbeitet sie danach unverändert. Fertige Varianten werden aus `2_FERTIG`
angeboten.
