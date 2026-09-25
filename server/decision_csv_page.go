package server

const decisionCSVPage = `<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSV klassifizieren · GopherLLM</title>
<style>
:root{color-scheme:light dark;font:16px/1.5 system-ui,sans-serif}body{max-width:760px;margin:40px auto;padding:0 20px}h1{font-size:2rem;line-height:1.2}p{color:light-dark(#485360,#b8c5d2)}form{display:grid;gap:20px}label{display:grid;gap:6px;font-weight:600}input,select,textarea,button{font:inherit;padding:10px;border:1px solid #8493a1;border-radius:6px;max-width:100%;box-sizing:border-box}textarea{min-height:85px;resize:vertical}small{font-weight:400;color:light-dark(#485360,#b8c5d2)}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}.check{display:flex;align-items:center;gap:10px;font-weight:400}button{cursor:pointer;background:#146248;color:white;border:0;font-weight:600}button:disabled{opacity:.6;cursor:wait}#status{white-space:pre-wrap}a{color:light-dark(#086046,#78dcbb)}[hidden]{display:none!important}@media(max-width:520px){.row{grid-template-columns:1fr}body{margin:24px auto}}
</style></head><body>
<p>GopherLLM · Native Klassifikation</p><h1>Eine Frage. Jede CSV-Zeile.</h1>
<p>Lade eine CSV hoch und beschreibe, was Laya pro Datensatz beurteilen soll. Das Ergebnis wird als letzte Spalte angehängt. Die Datei wird vom lokal laufenden Modell verarbeitet.</p>
<form id="form">
<label>CSV-Datei<input id="file" type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" required><small>Maximal 64 MiB Upload. Kopfzeile wird standardmäßig als Spaltenname verwendet.</small></label>
<label>Aufgabe / Frage<textarea id="instruction" required placeholder="Welcher Bereich soll diese Anfrage bearbeiten?"></textarea></label>
<div class="row"><label>Ergebnistyp<select id="type"><option value="choice">Kategorie (choice)</option><option value="noul">Ja/Nein-Wahrscheinlichkeit (noul)</option><option value="score">Bewertung auf einer Skala (score)</option></select></label>
<label>Trennzeichen<select id="delimiter"><option value=",">Komma</option><option value=";">Semikolon</option><option value="tab">Tabulator (TSV)</option></select></label></div>
<label id="criteria-label">Kategorien / Skalenstufen als JSON<textarea id="criteria">["Abrechnung", "Technik", "Vertrieb"]</textarea><small>Eine Liste oder bei Kategorien ein Objekt mit Beschreibungen: {"Abrechnung":"Zahlungen und Erstattungen"}</small></label>
<div class="row"><label>Eingabespalte<input id="column" placeholder="z. B. text"><small>Leer: kompletter Datensatz. Ohne Kopfzeile: Spaltennummer ab 1.</small></label>
<label>Neue Ergebnisspalte<input id="result-column" value="result" required></label></div>
<label class="check"><input id="no-header" type="checkbox">Datei enthält keine Kopfzeile</label>
<label class="check"><input id="result-json" type="checkbox">Vollständige Antwort mit Wahrscheinlichkeiten als JSON in der Ergebnisspalte</label>
<button id="submit" type="submit">CSV klassifizieren</button>
<button id="cancel" type="button" hidden>Abbrechen</button>
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
