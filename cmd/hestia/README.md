# Hestia

Die [native Hestia-App für macOS](../../apps/HestiaMac/README.md) nutzt
denselben Assistentenkern über Swift. Die folgenden Befehle starten die
separate Browser-Demo.

Hestia ist ein lokaler Sprach- und Textassistent. Er schaltet und dimmt
Lampen, beantwortet Fragen mit einem optionalen lokalen Chatmodell,
durchsucht die eigene Wissensdatenbank und kann ausdrücklich
angeforderte Wikipedia- und Websuchen mit Quellen beantworten. Die
Standardkonfiguration startet im Textmodus mit simulierten Lampen.

## Start

Hestia ist ein separates Go-Modul. Es benötigt Go 1.27.3 oder neuer und einen
tinyRAG-Checkout neben GopherLLM. Für persistente Wissensdatenbanken muss
dieser Checkout WAL-Datenbanken über `WALManager.Checkpoint` speichern.
Hestia nutzt tinySQL v0.74.0.

```text
workspace/
  GopherLLM/cmd/hestia/
  tinyRAG/
```

```sh
cd cmd/hestia
go test ./...
go run .
```

Öffne [Hestia auf diesem Rechner](http://127.0.0.1:8787). Probiere
„Schalte die Stehlampe im Wohnzimmer ein“ oder
„Dimme die Stehlampe im Wohnzimmer auf 30 Prozent“. Bei mehreren passenden
Geräten fragt Hestia nach. Gruppenaktionen brauchen eine Bestätigung.

## Sprachmodelle und Aktivierungswort

Voxtral, Phonon und Whisper bleiben lokal geladen. Sie nutzen denselben Sprachkern
wie die Swift-Bindung.

```sh
go run . -voxtral-model /pfad/zu/voxtral-realtime.gguf
go run . -phonon-model /pfad/zu/model.fermion
go run . -whisper-model /pfad/zu/ggml-base.bin -whisper-language de
```

Auf Apple Silicon aktiviert `go run -tags metal . …` die verfügbaren
Metal-Pfade für Voxtral und lokale Chatmodelle. Für ein entpacktes
Phonon-Modell liegt `config.json` neben `model.fermion`.
[Phonon-Modellformat](../../docs/phonon.md)

Whisper verwendet `-whisper-language auto` als Voreinstellung. Im JSON
wählst du `"backend": "whisper"`, setzt `model_path` auf das lokale
GGML-F16/F32-Modell und kannst `"language": "de"` ergänzen.
[Native Whisper-Inferenz](../../docs/generation.md#whisper-transcription)

Klicke **Aufnehmen** für einen einzelnen Befehl oder **Zuhören aktivieren**
für den Aktivierungsmodus. Sage das Wort und den Befehl in einem Satz,
etwa „Computer, schalte die Stehlampe im Wohnzimmer ein“. Ändere das Wort
über `-listen-word "Hallo Nova"` oder `listen_word` in der Konfiguration.
Ein leerer Wert deaktiviert den Aktivierungsmodus. Die Erkennung hängt vom
Sprachmodell, der Aussprache und dem Mikrofon ab. Beim Tabwechsel stoppt
das Zuhören. Die optionale Sprachausgabe nutzt eine als lokal gemeldete
deutsche Systemstimme des Browsers.

## Wiki, Web und Dokumente

Wikipedia und Websuche sind einzeln zuschaltbar. Hestia sendet bei einem
expliziten Suchbefehl den Suchbegriff an den jeweiligen Dienst. Wikipedia
erhält dabei die üblichen Netzwerkdaten. Die SearXNG-Instanz kann die
Anfrage an ihre konfigurierten Suchmaschinen weiterleiten. Trefferseiten
werden geöffnet, wenn du im Browser einen Quellenlink anklickst.

```json
{
  "listen_word": "Computer",
  "speech": {
    "backend": "phonon",
    "model_path": "/pfad/zu/model.fermion"
  },
  "research": {
    "wikipedia": true,
    "wiki_language": "de",
    "web_endpoint": "http://127.0.0.1:8888/search"
  }
}
```

Speichere die Konfiguration z. B. als `hestia.json` und starte
`go run . -config hestia.json`. Bei Voxtral oder Phonon verwende
`"backend": "voxtral"` bzw. `"backend": "phonon"` und `model_path`.
Lass `web_endpoint` weg, wenn du nur Wikipedia möchtest. SearXNG benötigt
`json` unter `search.formats`. Für entfernte Instanzen ist HTTPS erforderlich.

- **Wikipedia** – „Wikipedia: Ada Lovelace“ oder „Suche auf Wikipedia nach Ada Lovelace“.
- **Web** – „Websuche: Fahrplan Hamburg“ oder „Suche im Internet nach Fahrplan Hamburg“.
- **Eigene Dokumente** – Eine bestehende Hestia-Wissensdatenbank unter
  `data_dir/knowledge.db` lässt sich mit `-embed-model /pfad/zu/embedding.gguf`
  lokal durchsuchen. Beispiel: „Was sagen meine Unterlagen über Resturlaub?“

[SearXNG Search API](https://docs.searxng.org/dev/search_api.html) ·
[MediaWiki REST API](https://www.mediawiki.org/wiki/API:REST_API/Reference#Search_pages)

Für allgemeine Fragen kannst du mit `-chat-model /pfad/zu/chat.gguf` oder
`chat_model_path` ein lokales Chatmodell wie Ministral laden. Das Modell
bekommt die aktuelle Frage. Die Antwort wird als Text angezeigt und kann
bei aktivierter Sprachausgabe vorgelesen werden. Modellantworten können
keine Geräteaktionen auslösen.

## Echte Lampen über Home Assistant

Setze `HESTIA_HA_TOKEN` auf ein Home-Assistant-Zugriffstoken und ergänze
`home_assistant` in der JSON-Konfiguration. Mit diesem Eintrag ersetzt die
explizite Geräteliste die Demo-Lampen. Verwende die tatsächlichen
`light.*`-IDs aus Home Assistant. Hestia prüft nach einem Schaltbefehl den
gemeldeten Zustand.

```json
{
  "home_assistant": {
    "endpoint": "https://homeassistant.example",
    "lights": [
      {"id": "light.stehlampe", "name": "Stehlampe", "room": "Wohnzimmer"}
    ]
  }
}
```

Eine auf demselben Rechner laufende Instanz kann auch per Loopback-HTTP
erreicht werden, etwa unter `http://127.0.0.1:8123`. Der Adapter unterstützt
Lampen. Die Einbindung von HomeKit-Geräten erfolgt bei Bedarf in Home
Assistant. [Home Assistant REST API](https://contributors.home-assistant.io/core/internals/rest-api/) ·
[HomeKit Device](https://www.home-assistant.io/integrations/homekit_controller/)

## Datenschutz

- Die Weboberfläche und API sind nur über Loopback erreichbar. Die API
  akzeptiert gleichursprüngliche Browserzugriffe und lokale Clients. Sie
  vertraut den Benutzern und Programmen mit Zugriff auf den Rechner.
- Das Mikrofon ist standardmäßig aus. Der Browser erkennt Sprechpausen und
  sendet kurze Sprachabschnitte an den lokalen Hestia-Prozess. Das gewählte
  lokale Sprachmodell prüft das Aktivierungswort. Hestia verarbeitet
  Rohaufnahmen im Arbeitsspeicher und legt dafür keine Dateien an.
- Der Befehlsverlauf liegt standardmäßig im RAM. Mit
  `"persist_history": true` wird der Verlauf einschließlich Befehlstext in
  `data_dir/journal.jsonl` gespeichert. `data_dir` ist standardmäßig
  `./hestia-data`. Eine bereits vorhandene Verlaufsdatei bleibt auf der Platte.
- Web, Wikipedia und echte Geräte sind ab Werk deaktiviert. Die
  Weboberfläche lädt ihre Assets vom lokalen Hestia-Prozess.

`go test ./...` im Hauptrepo überspringt dieses verschachtelte Modul.
Für Hestias Tests führe den Befehl zusätzlich in `cmd/hestia` aus.
