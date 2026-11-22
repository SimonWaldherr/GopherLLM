# Hestia für macOS

Hestia ist eine native SwiftUI-App für Apple Silicon ab macOS 14. Swift und
Hestia verwenden ein gemeinsames statisches XCFramework mit einer Go-Runtime.
Die App übernimmt Mikrofonzugriff und Bedienung. Hestia verarbeitet Befehle,
Bestätigungen, Dokumente, Websuche und Geräteaktionen im App-Prozess.

## Lokal bauen und starten

Xcode und Go 1.27.3 oder neuer müssen installiert sein. Der tinyRAG-Checkout
liegt wie beim [Hestia-CLI](../../cmd/hestia/README.md) neben GopherLLM.
Voxtral, Phonon und Whisper laufen über GopherLLMs eigene Modell-Engine im
App-Prozess.

```sh
# Aus dem GopherLLM-Verzeichnis
bash apps/HestiaMac/build.sh build
open apps/HestiaMac/build/DerivedData/Build/Products/Debug/Hestia.app
```

Das Skript erstellt ein separates, generiertes Swift-Paket unter
`LinkedGopherLLM/`. Der Standardbuild der GopherLLM-Bindings für macOS und
iOS wird daneben weiter unterstützt. Der Debug-Build ist ad hoc signiert.
`build.sh prepare` aktualisiert das Paket für die Arbeit direkt in Xcode.

Die App startet im Textmodus mit simulierten Lampen. Unter **Einstellungen**
kannst du ein Modell auswählen und von Hugging Face laden oder eine lokale
Datei importieren. Für ein entpacktes Phonon-Modell wählst du den Ordner mit
`model.fermion` und `config.json`. Für einen direkten Download müssen beide
Dateien im selben Ordner des Hugging-Face-Repos liegen. Für Whisper wählst
du eine GGML-F16/F32-Datei wie `ggml-base.bin` oder gibst ihren Dateinamen
im Hub-Repo an. Whisper erkennt die Sprache in der App automatisch.
Das App-Bundle enthält keine Modellgewichte.

**⌘⇧R** startet und beendet eine einzelne Aufnahme im Hestia-Fenster.
**Zuhören** erkennt kurze Sprachabschnitte und prüft das eingestellte
Aktivierungswort. Sage das Wort und den Befehl im selben Satz. Beim Wechsel
in eine andere App stoppt die Aufnahme. Mikrofonzugriff benötigt die
Freigabe von macOS.

Embedding- und Chatmodelle lassen sich als lokale GGUF-Dateien importieren.
Wende anschließend die aktualisierte Konfiguration an. Mit einem
Embedding-Modell kannst du im Assistenten UTF-8-Dokumente indexieren und
durchsuchen. Für Wikipedia, SearXNG und Home Assistant gelten dieselben
[JSON-Einstellungen](../../cmd/hestia/README.md) wie beim CLI. Gib Token in
die dafür vorgesehenen Felder ein. Die App speichert sie im Schlüsselbund.

## Mac App Store

Der Xcode-Target enthält App Sandbox, Mikrofonfreigabe, Icon und ein
Datenschutzmanifest. Das Skript sammelt die Lizenzhinweise der gebauten
Bibliotheken und Go-Module im Bundle. Prüfe vor der Einreichung die Rechte
an der ausgelieferten Kombination und die App-Privacy-Angaben für deine
Store-Konfiguration. Für den Store-Eintrag ist eine öffentliche URL zur
[Datenschutzerklärung](PRIVACY.md) nötig.

```sh
DEVELOPMENT_TEAM=YOUR_TEAM_ID bash apps/HestiaMac/build.sh archive
```

Die Bundle-ID ist im Xcode-Target vorkonfiguriert und kann mit
`PRODUCT_BUNDLE_IDENTIFIER=…` beim Archivieren überschrieben werden.
Für den Upload öffnest du das signierte Archiv im Xcode Organizer und
wählst **Distribute App**. Ein App-Store-Connect-Eintrag mit passender
Bundle-ID und ein Apple-Developer-Team mit Distributionsberechtigung sind
dafür erforderlich. [Builds in App Store Connect hochladen](https://developer.apple.com/help/app-store-connect/manage-builds/upload-builds/)
