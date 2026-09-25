package server

const decisionCSVPage = `<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSV klassifizieren · GopherLLM</title>
<style>
:root { color-scheme:light dark; font:16px/1.5 system-ui,sans-serif; }
body { max-width:840px; margin:40px auto; padding:0 20px; }
h1 { font-size:clamp(1.8rem,5vw,2.6rem); line-height:1.15; margin:.3em 0; }
p, small { color:light-dark(#485360,#b8c5d2); }
.eyebrow { font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.8rem; }
form { display:grid; gap:24px; }
fieldset { min-width:0; margin:0; padding:20px; border:1px solid light-dark(#cbd5df,#47515d); border-radius:12px; }
legend { padding:0 8px; font-weight:700; }
.fields, label { display:grid; gap:8px; align-content:start; }
.fields { gap:18px; }
label { font-weight:600; min-width:0; }
input, select, textarea, button { font:inherit; padding:10px; border:1px solid #8493a1; border-radius:6px; width:100%; box-sizing:border-box; }
textarea { min-height:96px; resize:vertical; }
small { font-weight:400; }
.row { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }
.check { display:flex; align-items:center; gap:10px; font-weight:400; }
.check input { width:auto; }
.actions { display:flex; gap:12px; flex-wrap:wrap; }
button { cursor:pointer; background:#146248; color:white; border:0; font-weight:600; width:auto; }
button:disabled { opacity:.6; cursor:wait; }
#cancel { background:light-dark(#e4eaf0,#36414e); color:inherit; }
#status { white-space:pre-wrap; }
#status:empty { display:none; }
a { color:light-dark(#086046,#78dcbb); }
#download { display:inline-block; font-weight:700; padding:12px 0; }
.example { padding:12px 16px; background:light-dark(#f0f5f3,#1c2b26); border-radius:8px; }
.example code { overflow-wrap:anywhere; }
[hidden] { display:none!important; }
@media(max-width:520px) { .row { grid-template-columns:1fr; } body { margin:24px auto; } fieldset { padding:16px; } }
</style></head><body>
<p class="eyebrow">GopherLLM · Native Klassifikation</p><h1>Eine Frage. Jede CSV-Zeile.</h1>
<p>Lade eine CSV hoch und beschreibe, was Laya pro Datensatz beurteilen soll. Das Ergebnis wird als letzte Spalte angehängt. Die Datei wird vom lokal laufenden Modell verarbeitet.</p>
<p class="example"><strong>So sieht das Ergebnis aus</strong><br><code>id,text</code> → <code>id,text,kategorie</code><br>Alle ursprünglichen Spalten bleiben erhalten. Pro Datensatz kommt genau ein Ergebnis hinzu.</p>
<form id="form">
<fieldset><legend>1 · Datei auswählen</legend><div class="fields">
<label>CSV-Datei<input id="file" type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" required><small>Maximal 64 MiB Upload. Kopfzeile wird standardmäßig als Spaltenname verwendet.</small></label>
</div></fieldset>
<fieldset><legend>2 · Aufgabe festlegen</legend><div class="fields">
<label>Aufgabe / Frage<textarea id="instruction" required placeholder="Welcher Bereich soll diese Anfrage bearbeiten?"></textarea></label>
<div class="row"><label>Ergebnistyp<select id="type"><option value="choice">Kategorie (choice)</option><option value="noul">Ja/Nein-Wahrscheinlichkeit (noul)</option><option value="score">Bewertung auf einer Skala (score)</option></select></label>
<label>Trennzeichen<select id="delimiter"><option value=",">Komma</option><option value=";">Semikolon</option><option value="tab">Tabulator (TSV)</option></select></label></div>
<label id="criteria-label">Kategorien / Skalenstufen als JSON<textarea id="criteria">["Abrechnung", "Technik", "Vertrieb"]</textarea><small>Eine Liste oder bei Kategorien ein Objekt mit Beschreibungen: {"Abrechnung":"Zahlungen und Erstattungen"}</small></label>
</div></fieldset>
<fieldset><legend>3 · Ein- und Ausgabe</legend><div class="fields">
<div class="row"><label>Eingabespalte<input id="column" placeholder="z. B. text"><small>Leer: kompletter Datensatz. Ohne Kopfzeile: Spaltennummer ab 1.</small></label>
<label>Neue Ergebnisspalte<input id="result-column" value="result" required></label></div>
<label class="check"><input id="no-header" type="checkbox">Datei enthält keine Kopfzeile</label>
<label class="check"><input id="result-json" type="checkbox">Vollständige Antwort mit Wahrscheinlichkeiten als JSON in der Ergebnisspalte</label>
</div></fieldset>
<div class="actions"><button id="submit" type="submit">CSV klassifizieren</button>
<button id="cancel" type="button" hidden>Abbrechen</button></div>
</form>
<p id="status" role="status" aria-live="polite"></p><a id="download" download="classified.csv" hidden>Ergebnis-CSV herunterladen</a>
<script>
const el=id=>document.getElementById(id);let controller,downloadURL;
el('type').addEventListener('change',()=>{el('criteria-label').hidden=el('type').value==='noul';});
el('cancel').addEventListener('click',()=>controller?.abort());
el('form').addEventListener('submit',async event=>{
 event.preventDefault();el('download').hidden=true;if(downloadURL){URL.revokeObjectURL(downloadURL);downloadURL=null;}
 const file=el('file').files[0];if(!file)return;
 if(file.size>64*1024*1024){el('status').textContent='Die Datei überschreitet das Upload-Limit von 64 MiB.';return;}
 const question={type:el('type').value,instructions:el('instruction').value};
 try{if(question.type!=='noul')question.criteria=JSON.parse(el('criteria').value);}catch{el('status').textContent='Bitte gültiges JSON für die Kategorien oder Skalenstufen angeben.';return;}
 const options={question,input_column:el('column').value,result_column:el('result-column').value,delimiter:el('delimiter').value==='tab'?'\t':el('delimiter').value,no_header:el('no-header').checked,result_json:el('result-json').checked};
 const data=new FormData();data.append('options',JSON.stringify(options));data.append('file',file);
 controller=new AbortController();el('submit').disabled=true;el('cancel').hidden=false;el('status').textContent='Datensätze werden nacheinander klassifiziert …';
 try{
  const response=await fetch('v1/systemone/csv',{method:'POST',body:data,signal:controller.signal});
  if(!response.ok){let error;try{error=await response.json();}catch{}throw new Error(error?.error?.message||'Anfrage fehlgeschlagen (HTTP '+response.status+').');}
  const blob=await response.blob();downloadURL=URL.createObjectURL(blob);el('download').href=downloadURL;el('download').hidden=false;
  el('status').textContent=response.headers.get('X-Processed-Rows')+' Datensätze verarbeitet. Die Ergebnisdatei ist bereit.';
 }catch(error){el('status').textContent=error.name==='AbortError'?'Verarbeitung abgebrochen.':error.message;}
 finally{controller=null;el('submit').disabled=false;el('cancel').hidden=true;}
});
</script></body></html>`
