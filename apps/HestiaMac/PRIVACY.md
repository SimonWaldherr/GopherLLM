# Datenschutz in Hestia für macOS

Hestia verarbeitet Sprache, lokale Modellantworten und Dokumente auf deinem
Mac. Die App verwendet keine Werbung, Analyse oder Telemetrie. Das Mikrofon
startet erst, wenn du eine Aufnahme oder den Zuhörmodus aktivierst. Aufnahmen
liegen während der Verarbeitung im Arbeitsspeicher.

Deine Einstellungen und heruntergeladenen Modelle liegen lokal. Importierte
Dokumente werden für die Suche in einer lokalen Wissensdatenbank indexiert.
Der Befehlsverlauf liegt standardmäßig im Arbeitsspeicher. Wenn du die
Verlaufsspeicherung selbst aktivierst, speichert Hestia den Verlauf lokal.
Optionale Zugriffstoken liegen im macOS-Schlüsselbund.

Bei einem Modelldownload erhält Hugging Face die gewählte Modellanfrage und
gegebenenfalls dein Zugriffstoken. Wenn du externe Suche einschaltest und
einen Suchbefehl gibst, erhält der gewählte Dienst den Suchbegriff. Eine
konfigurierte SearXNG-Instanz kann ihn an ihre Suchmaschinen weiterleiten.
Home Assistant erhält Geräteabfragen, die ausgewählte Aktion und den dafür
konfigurierten Zugriffstoken. Diese Dienste erhalten keine Rohaufnahmen oder
lokalen Dokumente von Hestia. Ihre eigenen Datenschutzregeln gelten für die
Anfragen, die sie erhalten.

Du kannst die Mikrofonfreigabe in den macOS-Systemeinstellungen entziehen.
Externe Dienste lassen sich in Hestia deaktivieren. Entferne gespeicherte
Token, indem du die Token-Felder leerst und die Einstellungen anwendest.
Lokale Modelldateien, Wissensdatenbank und dauerhafter Verlauf liegen im
Hestia-Ordner unter „Application Support“ im App-Container.

Projekt und Kontaktmöglichkeiten findest du im
[GopherLLM-Repository](https://github.com/SimonWaldherr/GopherLLM).
